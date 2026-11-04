# How the brain settles and learns

This page covers the mechanics below [`Brain.compose`](tune.md), for building or debugging a
`NeuralGraph` or `Learner` directly. Learning rules live in
[learning](#learning-the-freenudged-rule) and [reward](reward.md);
[orientation](../README.md#where-the-learning-signal-comes-from) says where each family's
teaching signal comes from.

## Neurons and synapses

A neuron holds a potential `v` and publishes an activation `s`: a graded (rate) unit for
one neuron or, at coarse resolution, a population. A directed synapse carries one neuron's
activation to another. A `Connectome` declares the synapses and named populations:
`input` neurons receive the stimulus, `output` neurons are read and taught, `hidden`
neurons carry the rest (`cd.layered` creates all three). With `Learner(reciprocal=True)`,
the default, a synapse and its reverse share one learned weight.

On each step every neuron collects its synaptic input and then updates its own state:

```text
synaptic_input[i] = sum(weights[e] * s[pre[e]] for synapses e ending at i)
v[i] += dt * (synaptic_input[i] + drive[i] + bias[i] - strength * a[i] - v[i])
s[i]  = activation(v[i])
```

The effective weight of synapse `e` is `gain * count[e] * efficacy[e] * exp(log_gain[pre[e]])`.

| `NeuralGraph` parameter | Shape | Meaning |
|---|---|---|
| `efficacy` | one per synapse | Signed strength; starts from the connectome's signs and is what learners change |
| `log_gain` | one per neuron | Scales every outgoing synapse of that neuron; set it per cell class to declare a region's gain (`g = np.zeros(C.n); g[C.populations["ln"]] = -1.0`, then `log_gain=g`); learners leave it alone |
| `bias` | one per neuron | Constant drive; starts at zero |
| `count` | on the connectome | Number of synaptic contacts |
| `gain` | on the `NeuronModel` | Global scale |

The graph's parameter arrays are read-only views; replace one by assigning a whole array
(`graph.bias = b`) or with `graph.with_parameters(...)`, which rebuilds the effective
weights. Backend, device, precision and `dense_limit`: see [backends](backends.md).

The drive is added to a neuron's input on every step. It does not clamp the potential:
synaptic input, bias and adaptation move a stimulated neuron too.
`NeuralGraph.settle(stimulus)` accepts a dense vector, a list of neurons (each driven at
`stimulus_amplitude`) or a `{neuron: level}` map (`level * stimulus_amplitude`).

A `Nudge` adds teaching drive `beta * mask * (target - s)`.

- With `softmax_temperature` set, the masked neurons compete and the drive is
  `beta * (target - softmax(s / T))`, a cross-entropy nudge. `groups` (one integer per neuron,
  `-1` for none) splits them into separate softmax groups; a temperature vector must be
  constant within each group.
- `weight`, one number per batch row, scales the nudge per row. A negative weight pushes away
  from the target, which is how an advantage enters reward learning.
- `anchor` and `anchor_gain`, supplied together, add a boundary drive
  `anchor_gain * (anchor - s)` that `beta`, `mask` and `weight` do not scale, so positive and
  negative phases share the same boundary. `anchor_gain` is a nonnegative vector over neurons.

In the graph update a neuron reads only its own state, its synaptic input and its drive. Two
operations read a declared group: a softmax nudge reads its group, and key normalisation in
`FastSynapses` (`normalize=True` or `rule="delta"`) reads the whole key vector
([standalone memory stores](memory.md#reading-and-writing-directly)).

## Settling and equilibrium

`settle(stimulus, *, steps=60, state=None, mask=None, trajectory=False, nudge=None,
tolerance=None)` and `settle_batch(drive, ...)` (dense `(batch, n)` drive, same keywords)
run the update from rest or from a supplied `state`. `steps` caps the work; `tolerance`
stops early once the largest activation change in a step falls below it (`None` or zero
disables this). `mask` holds fractions in `[0, 1]`, shape `(n,)` or `(batch, n)`, and zeros
ablated neurons; pass the same `mask` and `nudge` to a settle and to its check. The
returned `BrainState` holds `v`, `activation`, `adaptation`, `steps` taken,
`activity_change` (total movement per row) and, with `trajectory=True`, the activation
after every step. It can be a transient, a fixed point or part of an oscillation.

`residual(drive, state)` returns the largest fixed-point equation error per row. Check it
instead of movement, since a saturated neuron moves little while its equation is far from
satisfied. `equilibrate(drive, *, budget=512, chunk=32, tolerance=1e-5, state=None,
mask=None, nudge=None, damping=0)` continues until every row's residual is below `tolerance`
or the budget is spent, at one extra transport per chunk (on the compiled CPU path, the block
products of the ranges the chunk's last sweep moved). All rows advance together. It
returns an `Equilibrium` with `state`, per-row `residual`, `steps` (sweeps in this call) and
`converged` (residual within tolerance), which `qualified` repeats. `damping=k` splits the
budget over up to `k + 1` attempts, each at half the previous `dt`, all checked against the
original equations. Report a spent budget and act on it.

Regions that share one connectome settle toward one joint fixed point under the current
drive; independent solves merged afterwards do not couple them. An equilibrium need not be
unique or stable. Carrying a state between inputs (warm start) can save work or select a
different attractor than starting from rest (cold start); test both. A unique attracting
equilibrium forgets its initial condition, so history needs a trace, a memory or a drive
that carries it.

When every neuron's absolute incoming effective weight sum is below `1 / L` (`L` the
largest activation slope) and adaptation is off, one free-phase step (no nudge) is a
contraction and [`cd.certificate(brain)`](#the-settling-certificate) bounds the distance to the unique
equilibrium.

## Activation and adaptation

`NeuronModel(dt=0.2, slope=4.0, threshold=1.5, gain=0.02, stimulus_amplitude=3.0,
adaptation=None, leak=0.0)` rebases a sigmoid so zero potential emits zero; with zero
drive, bias and adaptation the all-zero state is an exact fixed point. `dt` is the
relaxation fraction in `(0, 1]` and `steps` counts updates of the map; neither is elapsed
time. `slope` and `gain` must be positive and `leak` lies in `[0, 1]`. With `leak` above
zero a neuron below rest emits a small negative activation, no lower than `-leak`, so a
learning rule can tell a silent neuron from a dead one.

Use `learning_neuron_model(gain=1.0, *, slope=1.0, leak=0.1, dt=0.5, stimulus_amplitude=1.0)`
for nets that learn. It sets threshold zero; unit slope keeps a fan-scaled layer responsive, a
unit stimulus reads input levels roughly linearly, and the larger `dt` converges in tens of
steps. Builders start with zero bias, so many hidden neurons can sit near rest under signed
inputs; measure activation distributions on your stream and shift `bias` if needed.

`Adaptation(tau_steps=50.0, strength=1.0)` adds one variable per neuron,
`a <- a + (s - a) / tau_steps`, and subtracts `strength * a` from its drive. In mutually
inhibitory circuits it produces rhythm instead of a fixed point. It is off by default. Leave
it off for equilibrium learning; the certificate excludes it.

## Three state lifetimes

| State | What changes it | How to manage it |
|---|---|---|
| Potential, activation, adaptation | Settling steps | Pass `state=` to continue; omit it to start from rest |
| Trace or fast-memory matrix | Explicit activity or observation updates | Keep it through a continuing stream; reset only at a declared terminal |
| Weights, biases, consolidated associations | Learner updates, reward, `SynapticMemory.observe` | Save each owning component; evaluate on a separate snapshot |

A raw `NeuralGraph` changes no weights by settling. `Brain.step` schedules learning in a
composed brain; a custom composition owns its update calls. See
[traces](api.md#streams-cadencestream), [memory](memory.md) and
[standalone stores](memory.md#standalone-memory-stores).

## Three clocks

| Clock | What changes | What stays fixed within the operation |
|---|---|---|
| Environment step | Observations, executed actions and carried state | Previously admitted evidence |
| Inference iteration | The interpretation of the same evidence | Evidence, parameters and the memory snapshot |
| Learning update | Parameters, memory contents or a learned address map | Which targets were observed, taught or imagined |

Assign every state update to one clock. An environment step has the duration its simulator
or body declares; solver sweeps and `dt` have none, so measure wall time separately. An
imagined future has its own step index inside a private branch; its steps simulate
environment time and count neither as inference on the present nor as new observations.
[Continuous interaction](start.md#defaults-and-the-thinking-clock) states which events
advance trace, eligibility, arousal and learning in `Brain`.

A previous-state mixing term adds no inference depth to a one-step read from a zero initial
state, because that term is zero. More internal steps or a deeper port are a separate
computation with a separate cost.

## Prediction error is not model error

A correct probabilistic model cannot predict which random outcome occurs. A fair-coin model
assigns one half to heads; observing heads resolves this toss and gives no reason to switch
to an always-heads rule. An observation can change the current belief and the best action
while the learned dynamics stay as they are, because state estimation and model learning are
separate operations. The cross-entropy logit gradient `p - y` is nonzero on each toss and has
zero expectation under a calibrated prediction, so sample updates average out. Change the
model for repeated systematic errors. Judge probabilistic predictions by mean log loss and
calibration against the declared uncertainty, and do not widen uncertainty to make every
error look acceptable.

An unexpected observation supplies evidence and possibly a teaching target. The `beta` of a
learning contrast, inference damping and the random drive below are separate quantities.

<a id="detuning-inside-a-life"></a>

## Sample candidate equilibria by drive perturbation

A settled graph gives one answer under one drive. To sample alternatives inside one
decision, repeat the drive in a batch, add a bounded random drive to the latent neurons of
each row, settle with `equilibrate`, keep the converged rows and rank their actions by a
validated read such as a table of witnessed outcomes, a
[delta-rule store](memory.md#standalone-memory-stores) or a critic. Only this decision's
drive changes; parameters, memories and the genome stay as they are. This perturbation is
separate from the learning nudge (`beta`).

```python
import numpy as np
import cadence as cd

connectome = cd.layered(2, 16, 2, density=1.0, seed=1)    # cue in, action out
brain = cd.NeuralGraph(connectome, cd.learning_neuron_model())
cue_units, latent, action_units = (list(connectome.populations[k]) for k in ("input", "hidden", "output"))

witnessed = np.zeros((2, 2))             # witnessed: the action equal to the cue pays
for cue in range(2):
    for action in range(2):
        witnessed[cue, action] = float(action == cue)

cue = np.eye(2)[1]
drive = np.zeros((8, connectome.n))      # eight candidate rows, the same cue in each
drive[:, cue_units] = cue
drive[:, latent] += np.random.default_rng(0).uniform(-0.5, 0.5, (8, len(latent)))
result = brain.equilibrate(drive, budget=512, chunk=32, tolerance=1e-6)
candidates = result.state.activation[:, action_units].argmax(axis=1)
score = witnessed[1, candidates]         # the outcome each candidate was seen to bring
score[~result.converged] = -np.inf       # a row without an equilibrium offers no candidate
choice = int(candidates[np.argmax(score)])
print(candidates, score.round(2), choice)
```

A uniform draw in `[-a, a]` keeps each added drive within `a`; a Gaussian draw has no such
bound. In a composed brain build the repeated drive with `brain.stimulus([observation])`,
settle with `brain.brain.equilibrate`, perturb `brain.association_index` and read actions
at `brain.motor_index`; `brain.basal_ganglia.value(result.state)` ranks the same rows by the
critic. The ranking helps only as far as the table, store or critic has learned
consequences.

## Controls and when to add machinery

A [protocol](../benchmarks/instruments/README.md#protocols-held-out-tests-with-preconditions) declares stimuli, readouts,
interventions and predicates. A [shuffled connectome](../benchmarks/instruments/README.md#the-shuffled-control) tests
whether a response depends on the particular connections; gain selection can
[cap the active fraction](../benchmarks/instruments/README.md#gain-selection-under-a-sparsity-cap) so high gains do
not saturate an excitatory circuit. Both are benchmark instruments of a source checkout.
`cd.conformance(brain, stimulus)` compares a trajectory with the neuron-by-neuron
reference. A [receipt](evaluate.md#receipts) binds stored results to their sources.

Settling, memory and learning are separate operations because their state lifetimes
differ. A new mechanism needs a failure it fixes and an ablation on the same inputs,
budget and seeds, with its state and cost counted.

## Learning: the free/nudged rule

A Cadence brain learns by settling twice: once on the observation alone (the free phase) and
once with its outputs pulled toward the shown answer (the nudged phases). Every synapse then
moves on the difference between its own endpoint activities in the two. In a composed brain,
teach with `Brain.lesson`, or with `Brain.step(teacher=...)`, which gives the same lesson and
then acts. A standalone `Learner` runs the rule on any `NeuralGraph`:

```python
import numpy as np
import cadence as cd

brain = cd.Brain.compose(inputs=4, actions=2, seed=0)
cue = np.array([[1.0, 0.0, 0.0, 0.0]])
lesson = brain.lesson(cue, np.array([1]))           # teach, without acting
action = brain.step(cue, teacher=np.array([1]))     # the same lesson, then an act
print(lesson["accepted_presentations"], brain.last_learning["demonstration_total_steps"])

connectome = cd.layered(4, 16, 2, seed=0)
learner = cd.Learner(
    cd.NeuralGraph(connectome, cd.learning_neuron_model()), connectome.populations["output"]
)
drive = np.zeros((2, connectome.n))
drive[:, :4] = np.eye(4)[:2]
for _ in range(20):
    state, report = learner.step(drive, np.array([0, 1]))
print(learner.predict(drive), report["total_row_sweeps"])
remaining = learner.brain.residual(drive, state.free)  # equation residual per row
```

`Brain.step` keeps the lesson report in `last_learning` with a `demonstration_` prefix. See
[reward](reward.md) for reward learning, [memory](memory.md) for memory writes, and
[explicit learning with `step`](teach.md#explicit-learning-with-step) for teacher-acted moments.

### 1. What learning changes

Learning moves `efficacy` (one signed number per synapse) and `bias` (one per neuron); the
connectome and `log_gain` stay fixed. Call the effective weight of synapse `e`
([above](#neurons-and-synapses)) `W[e]`; in a `layered` graph with `learning_neuron_model()`
it equals `efficacy[e]`.

### 2. What settling computes

Each phase repeats the neuron update ([above](#neurons-and-synapses)), with weights `W` and
the phase's nudge added to the drive. With `learning_neuron_model()`,
`act(v) = tanh(v/2)` for `v >= 0` and `0.1 · tanh(v/2)` below.
Rest publishes 0, drive saturates near `v ≈ 4`, and the leak keeps a small negative response
the rule can move. A phase stops when no activation moves more than `tolerance` or at its step
cap. Check the equations with `brain.residual(drive, state)` (with the phase's `nudge` for a
nudged state), since saturation or a tiny `dt` can stall a neuron far from its fixed point.

### 3. The two phases and the update

**Free phase.** Settle under the stimulus alone to `s⁰`. The outputs are the brain's answer.

**Nudged phases.** From `s⁰`, settle with an extra drive on the output neurons:

    nudge[i] = beta · weight · (target[i] - p[i]),   p = softmax(s[outputs] / T)

This is a cross-entropy nudge. `weight` is 1 for a label and the advantage for a reward. Two
phases run from `s⁰`, at `+beta` (`s⁺`) and `-beta` (`s⁻`). With several output `slots`, each
slot is its own softmax and carries `beta / slots`.

**Update.** Every synapse reads its two endpoints in the two phases:

    contrast[e]  = mean_batch( s⁺[pre]·s⁺[post] - s⁻[pre]·s⁻[post] ) / (2 beta)
    bias_term[i] = mean_batch( s⁺[i] - s⁻[i] ) / (2 beta)
    efficacy[e] += eta · contrast[e];   bias[i] += eta_bias · bias_term[i]

Supplied states without the `-beta` phase (`update(free, nudged)`) contrast against `s⁰`, with
`beta` in place of `2 beta`. With `reciprocal=True` a synapse and its reverse share one efficacy
and move by their mean contrast. Plastic efficacies
are clipped to `±scale_cap`. No backward computation graph is kept.

### 4. One observation repairs a settled prediction

For a custom world model, save the free prediction (`Learner.free`) under the observation and
proposed action. Once the action has run, its observed consequence is the target, and both
nudged phases start from that saved state. The repair assigns no credit across delays; use
[reward eligibility](reward.md) for that. Score an outcome before teaching it. Where an outcome
follows its reading directly, a [delta-rule store](memory.md#standalone-memory-stores) learns it without settling
phases.

### 5. Why the contrast is a gradient

This section concerns the plain synaptic sum, without apical normalization.
Let `F` be the recurrent neurons with the input activations held fixed. With no adaptation,
symmetric effective recurrent weights and a monotone differentiable activation, the
continuous-time dynamics descend

    E(v_F) = Σ_{i∈F} ∫₀^{v[i]} u·act'(u) du − ½ Σ_{i,j∈F} W[i→j] s[i] s[j] − Σ_{i∈F} d[i] s[i]
    d[i] = stimulus[i] + bias[i] + Σ_{k input} W[k→i] s[k]

On a smooth stable branch, converged phases obey the
[equilibrium-propagation identity](https://arxiv.org/abs/1602.05179): the limiting contrast is
minus the loss gradient for the one weight of a reciprocal pair. The quadratic nudge's loss is
half squared error; the cross-entropy drive omits `1/T`, so its loss is `T` times cross-entropy.
The [centered estimator](https://arxiv.org/abs/2006.03824) leaves an error of order `beta²`, and
a phase solved to sup-norm error `eps` adds at most `A·(eps⁺ + eps⁻)/beta` (`A` an activity
bound), so a smaller nudge needs a tighter solve. A nudge across the kink at zero or into
another attractor loses this accuracy; `leak=1` gives the smooth `tanh(v/2)`.

`Learner.contrast` returns the statistic. Where `c[e] = gain · count[e] · exp(log_gain[pre[e]])`
is not one, multiply by `c[e]` to get minus the gradient for a tied `efficacy`, and divide by `T`
for ordinary cross-entropy. Tie groups (`Learner(tie_groups=...)`) and reciprocal pairs combine
transitively; a group moves by its members' mean contrast and frozen members stay fixed. Tying
constrains increments, so tied members stay equal only if they start equal.
`tests/test_equilibrium.py` checks the conversion by finite differences;
[benchmarks/local_equilibrium](../benchmarks/local_equilibrium/README.md) compares the update
with an implicit-gradient learner on the same network, and the
[certificate](#equilibrium-propagation-scope) says when phases contract.

#### Qualified teaching budgets

Teaching is finite by default: the contrast is read after at most `free_steps` and
`nudged_steps` sweeps. `qualified=True` requires every phase to meet the full potential and
adaptation residual within `tolerance`, with up to `damping` step halvings inside its budget.
A failed phase raises `LearningPhaseError` (`phase`, `phases`, `report`, `hint`) and writes
nothing.

Under `qualified=True` both budgets are settle budgets, and a nudged phase needs about as many
sweeps as the free one. Raise `nudged_steps` to match `free_steps`: at the composed
`free_steps=1024`, `nudged_steps=12` a qualified brain refuses lessons on realistic input.
`qualified=True` with `nudged_steps < free_steps` warns, and a refusal names the mismatch in
its `hint`. Opt in with named overrides, which keep the composed rates:

```python
checked = cd.Brain.compose(
    inputs=4, actions=2, seed=7, learning_qualified=True, learning_nudged_steps=1024,
)
qualified_lesson = checked.lesson(cue, np.array([0]))
print(qualified_lesson["qualified"], qualified_lesson["total_steps"])
```

Every qualified phase starts at the full step and takes the damping ladder from there
(`learning_damping` halvings at most). Ordinary acts and evaluation keep their own numerical
routes, and reward eligibility keeps its finite phase. A retired setting
is refused by name with a message saying what to use instead ([CHANGELOG](../CHANGELOG.md)).

Qualification certifies the equations at those states; the gradient reading also needs the
assumptions above. `contrast`, `update` and `apply` act on supplied states. Answers follow
[qualification and refusal](start.md#qualification-and-refusal).

#### Finite free phases that run out of budget

Finite teaching updates from whatever state the budgets reached. When a free phase uses all of
`free_steps`, usually without reaching the movement stop, `Learner.step` reports
`free_budget_exhausted` (`demonstration_free_budget_exhausted` through `Brain.step`) and warns;
`free_residual` holds the equation residual. Fixed-length phases (`tolerance=None`) stay silent. Raise `free_steps`,
lower the gain, or select `qualified=True` to refuse such lessons.

A lesson report holds `attempted_presentations`, `accepted_presentations`, `accepted`,
`qualified`, `qualification_required`, the mean absolute steps `scale_step` and `bias_step`,
and per phase (`free`, `nudged`, `opposite`) `_steps`, `_residual`, `_residual_checks`,
`_damping_halvings`, `_stagnation_checks` (complete-state comparisons that let a stalled
halving pass its unused sweeps on), `_row_sweeps` and `_row_residual_checks`, with `total_*`
sums. A refused lesson's work is in the error's `report`.

### Feedback and the reach of a nudge

A contrast is zero where neither endpoint moves under the nudge, so hidden neurons need a
feedback path from the outputs; `layered` supplies tied output-to-hidden feedback. Asymmetry
among free neurons breaks the energy argument; one-way projections from fixed inputs keep it.
`ep_structure(brain, fixed_inputs=...)` checks this structure and nothing about convergence.
Adding reverse synapses to a measured connectome changes the model ([connectomes](anatomy.md#brains-from-a-connectome)).

### 6. Features on input neurons

A batched drive has one column per neuron. `NeuralGraph.stimulus_levels(levels)` multiplies
dense, finite, possibly signed per-neuron levels by the neuron model's `stimulus_amplitude`; it
neither pads nor clips. `layered` input neurons have no inbound synapses, so with zero bias each
settles at `v = stimulus`:

```python
pattern = cd.layered(2, 8, 2, density=1.0, seed=0)
graph = cd.NeuralGraph(pattern, cd.learning_neuron_model())
levels = np.zeros((2, pattern.n))
levels[:, list(pattern.populations["input"])] = [[1.0, 0.0], [0.0, 1.0]]  # one row per case
pattern_drive = graph.stimulus_levels(levels)
```

Fit scaling and encodings on training data. The `Learner` operations:

| call | what it does |
|---|---|
| `step(drive, labels, warm=None, weight=None, *, answer=None)` | free phase, nudged phase(s), one update; returns `(LearnedState, report)`. `labels` holds one output index per row, `(batch, slots)` when slotted; `weight` scales each row's pull (an advantage) |
| `free(drive, warm=None)`, `nudged(drive, free, target, sign=1.0, weight=None)` | one phase; `sign=-1.0` is the opposite phase; `targets(labels)` builds the per-neuron target |
| `contrast(free, nudged, opposite=None)`, `contrast_rows(...)` | the per-synapse and per-neuron terms divided by `beta` or `2 beta`, batch mean or per row |
| `update(free, nudged, opposite=None)` | momentum, normalization and the step from supplied states |
| `apply(delta_scale, delta_bias)` | a precomputed step through masks, synapse rates, tying and the cap |
| `predict(drive)`, `accuracy(drive, labels, batch=256)` | most active output per slot after a free phase; fraction correct over rows and slots |

`LearnedState` holds the `free`, `nudged` and `opposite` states and `taught`. An update costs
three settles. A `warm` state starts the free phase from an earlier state; within one basin
this saves steps, across attractors it can change the answer. A finite warm phase that stops
at its step cap keeps its transients and need not be an equilibrium. Reset at independent
episode boundaries, and compare warm and cold settles on changing inputs.

#### Pattern targets

For continuous or multi-label targets, build one target column per neuron and run the phases
directly with the quadratic nudge (`beta · (target − s)`):

```python
regressor = cd.Learner(graph, pattern.populations["output"], cd.LearnerConfig(nudge="quadratic"))
target = np.zeros_like(pattern_drive)
target[:, regressor.output_index] = [[0.2, 0.8], [0.8, 0.2]]
free = regressor.free(pattern_drive)
plus = regressor.nudged(pattern_drive, free, target)
minus = regressor.nudged(pattern_drive, free, target, sign=-1.0)
regressor.update(free, plus, minus)
reading = regressor.free(pattern_drive).activation[:, regressor.output_index]
```

Keep targets inside the activation's range and read activations directly.

#### Two learners on one graph

An update replaces a learner's graph. Build a second learner with `Learner.sharing` so both read
one `GraphHolder`:

```python
shared = cd.layered(4, 8, 4, seed=1)
outputs = list(shared.populations["output"])
first = cd.Learner(cd.NeuralGraph(shared, cd.learning_neuron_model()), outputs[:2])
second = cd.Learner.sharing(first, outputs[2:])  # other constructor fields as keywords
shared_drive = np.zeros((1, shared.n))
shared_drive[:, :4] = 1.0
first.step(shared_drive, np.array([1]))
assert second.brain is first.brain and second.holder is first.holder
```

An update replaces `holder.brain` once every check passes; assigning `learner.brain` replaces
it for every learner on the holder. Each learner keeps its own optimizer history. Disjoint
`plastic_synapses` and `plastic_neurons` masks keep one's updates off the other's
([write a cortex](anatomy.md#which-synapses-a-head-owns)). `Learner.load` and
`dataclasses.replace` give a holder of its own; `copy.copy` shares it.

#### Calibrating the operating point

Calibrate on training inputs and freeze the result before evaluation. Two tools set the
operating point without teaching anything.

`learner.calibrate(drive, level=None, grid=None)` picks and installs the global synaptic gain
whose free outputs sit nearest a target. The default target puts the mean, over rows and
slots, of the highest output activation at 0.5. An explicit `level` targets the mean over
every output, which pushes all choices up together and can saturate the readout. The default
grid is the current gain, then its multiples `2**k` for `k = -8..8`, `k ≠ 0`; ties keep the
first. Under `qualified=True` candidates must meet the equations. `last_calibration` holds
every candidate's gap, residuals and work; with no usable candidate `calibrate` raises
`RuntimeError`. A refused candidate has `gap` `None`; its `admitted`, `finite`,
`activation_consistent` and residuals give the reason. The sampled gains may never reach the
target, and reaching it does not show that teaching will work.

`calibrate_bias(brain, drives, targets)` moves one population where the gain would scale
inhibitory synapses too. It bisects biases toward a mean activation in `[0, 1]` per
population (a name or neuron indices in `targets`), or per member with `per_neuron=True`, and
returns the bias array, leaving the graph unchanged. Each bias is bisected `iterations=16`
times within `span=(-6.0, 6.0)`, every population in turn, for `rounds=3`; each solve settles
for at most `steps=100` sweeps to `tolerance=1e-4`. An unreachable target ends at a span
endpoint, so check `final_means` and `target_gaps` in the dict passed as `report`. With
`qualified=True` (and up to `damping=3` halvings) every solve must meet the equations; a
refused or nonfinite solve raises `RuntimeError`.

```python
operating = cd.Brain.compose(4, 3, modules=(8,), lateral=0.0, seed=9)
training = operating.stimulus(np.array([[1.0, 0.2, 0.0, 0.1], [0.2, 1.0, 0.1, 0.0]]), memory=False)
calibration = {}
bias = cd.calibrate_bias(
    operating.brain, training, {"motor": 0.25},
    qualified=True, damping=3, steps=256, tolerance=1e-7, report=calibration,
)
candidate = operating.brain.with_parameters(bias=bias)
assert np.all(candidate.equilibrate(training, budget=256, tolerance=1e-7, damping=3).qualified)
operating.learner.brain = candidate
print(calibration["final_means"], calibration["target_gaps"])
```

### 7. Every knob

`LearnerConfig()` holds the standalone defaults below. In a composed brain each field is the
`learning_<field>` override of `compose` and `retune`, except `eta`, `eta_bias`, `normalize`
and `normalize_floor`, which the teaching point sets (`teaching="founder"`, `"normalized"` or
`"deep"`); the composed founders are in
[defaults and expert overrides](tune.md#defaults-and-expert-overrides).
`learning=LearnerConfig(...)` replaces the composed config whole.

A standalone learner's config is frozen; change it between updates with
`learner.config = dataclasses.replace(learner.config, eta=...)`. `eta_bias=None` derives
`eta / 10`; a resolved value is kept when `eta` changes unless set again or passed as `None`.
Select rates on development data and freeze them.

| knob | where | what it does | default |
|---|---|---|---|
| `beta` | `LearnerConfig` | nudge strength; smaller is closer to the gradient | 0.1 |
| `eta`, `eta_bias` | `LearnerConfig` | efficacy and bias steps; an `eta_bias` above `eta` warns | 0.2, `None` (`eta / 10`) |
| `free_steps`, `nudged_steps` | `LearnerConfig` | sweep caps of the free and nudged phases; [settle budgets](#qualified-teaching-budgets) when qualified | 100, 50 |
| `tolerance` | `LearnerConfig` | movement stop for finite phases, equation residual when qualified; `None` fixes the length | 1e-4 |
| `nudge` | `LearnerConfig` | `"cross_entropy"` or `"quadratic"`, the latter for a standalone learner of pattern targets; a composed brain refuses it | `"cross_entropy"` |
| `temperature` | `LearnerConfig` | softmax temperature `T` of the cross-entropy nudge; in a composed brain also the sampled policy's base temperature | 0.2 |
| `momentum` | `LearnerConfig` | in `[0, 1)`: step on a bias-corrected running mean of each contrast | 0.0 |
| `normalize`, `normalize_floor` | `LearnerConfig` | forgetting factor in `[0, 1)` of a running RMS that divides each step, plus a floor ([below](teach.md#rates-under-normalization)) | 0.0 (off), 1e-3 |
| `normalize_relative` | `LearnerConfig` | above zero, in place of `normalize_floor`, that share of the RMS of the synapse's projection (a bias: its range) ([rates](teach.md#rates-under-normalization)) | 0.0 (off) |
| `scale_cap` | `LearnerConfig` | largest plastic efficacy magnitude; smaller keeps readouts out of saturation | 8.0 |
| `qualified`, `damping` | `LearnerConfig` | require full-equation phases; most `dt` halvings per phase | `False`, 3 |
| `surprise` | `LearnerConfig` | [teach only differing rows](teach.md#lessons-only-where-the-answer-differs) | `False` |
| `surprise_share` | `LearnerConfig` | with `surprise`, scale a lesson's step by the share of its rows it taught ([in proportion to the surprise](teach.md#in-proportion-to-the-surprise)) | `False` |
| `rehearsal`, `rehearsal_capacity` | `LearnerConfig` | in a composed brain, after each lesson replay that many times its rows again, each with its episode, drawn from a reservoir of that many lesson rows ([rehearsal](teach.md#teach-several-types-in-sequence-rehearsal)) | 0.0 (off), 512 |
| `outputs`, `config` | `Learner` | output neuron indices; the `LearnerConfig` | required, `LearnerConfig()` |
| `plastic_synapses`, `plastic_neurons` | `Learner` | bool masks of moving efficacies and biases | all |
| `reciprocal` | `Learner` | tie each synapse and its reverse into one efficacy | `True` |
| `tie_groups` | `Learner` | int per synapse, `-1` untied; a group shares one step | `None` |
| `synapse_rate` | `Learner` | nonnegative step multiplier per synapse, before tying | `None` (1) |
| `slots` | `Learner` | outputs split into softmax groups: a count of equal groups or a size per group | 1 |
| `gain`, `slope`, `leak`, `dt`, `stimulus_amplitude` | `learning_neuron_model` | synaptic gain, activation slope, sub-rest response, integration step, input scale | 1.0, 1.0, 0.1, 0.5 (`Brain.compose` uses 1.0), 1.0 |
| `density`, `feedback`, `lateral`, `seed` | `layered` | input-to-hidden density, feedback scale, output-to-output weight, wiring seed | 0.3, 1.0, 0.0, 0 |
| `count`, `init`, `excitatory_forward` | `layered` | contacts per synapse, initial magnitude scale, positive forward signs only | 1.0, 1.0, `False` |
| `skip`, `skip_init` | `layered` | direct input-to-output synapses and their own init scale (`None` uses `init`) | `False`, `None` |

With `reciprocal=True` the learner refuses (`ValueError`) a connectome with two synapses between
the same pair of neurons; merge them with `Connectome.from_synapses` (counts add) or pass
`reciprocal=False`. The learner keeps `plastic_synapses` and `plastic_neurons` by reference, so
editing those arrays changes later updates. Tie groups are fixed at construction: rebuild the
learner to change them.

`embedded(vocabulary, positions, dim, hidden, outputs, seed=0, init=1.0)` returns a one-hot
token window through a shared embedding and its `tie_groups`. `Learner.parameters()` counts
plastic efficacies (a pair or group once) plus plastic biases. `updates` counts applied updates,
`contrast_updates` those from contrasts; `save` and `load` are in the [API reference](api.md).

## The settling certificate

`cd.certificate(brain)` gives a sufficient condition for a brain's free settling to converge
to exactly one equilibrium, and bounds the distance to it after a finite number of steps. It
costs one pass over the synapses.

### The rule

One settling step of a brain without adaptation applies the free map

```text
F_c(v) = (1 - dt) v + dt (W act(v) + c)
```

`W` holds the effective signed weights, contact counts and gains included; `c` holds the bias
and the current sensory, trace and associative-memory drive. Let `L` be the largest slope of
the activation (`cd.lipschitz_constant(model)`) and `rho` the largest absolute incoming
effective weight sum of any neuron (`cd.row_mass(brain)`). If `L rho < 1` and `0 < dt <= 1`,
each step shrinks the largest potential difference between two states by
`q = 1 - dt (1 - L rho) < 1`. By the Banach fixed-point theorem there is exactly one
equilibrium, and

- after `k` steps from any start the distance to it is at most `q^k / (1 - q)` times the first
  step's movement, and the last step's movement `m` bounds it by `m / (1 - q)`;
- after the drive changes by `delta` (largest change of any drive), a warm start from the old
  equilibrium is within `tol` of the new one after about
  `log(dt delta / ((1 - q) tol)) / log(1 / q)` steps, and needs none after no change;
- with no stimulus and no bias the equilibrium is rest;
- removing neurons moves the equilibrium by at most `dt` times the synaptic mass they sent,
  times the activation bound, over `1 - q`.

For the library's activation `L = (slope / 4) max(1 / (1 - rest), leak / rest)`, with `rest`
the rest emission. Under `learning_neuron_model()` (slope 1, threshold 0, leak 0.1) `L = 1/2`,
so a learning brain is certified when every incoming effective weight sum is below 2.
`certificate` raises `ValueError` for a model whose `dt` lies outside (0, 1].

### Reading it

```python
import cadence as cd

connectome = cd.layered(8, 12, 3, density=1.0, init=0.25)   # row mass 1.39: certified
brain = cd.NeuralGraph(connectome, cd.learning_neuron_model())
cert = cd.certificate(brain)
print(cert.rate, cert.certified, cert.mass_limit)   # q, the verdict, and 1 / L
print(cert.error_bound(1e-4))                       # distance left after a step that moved 1e-4
print(cert.apriori_bound(1.0, steps=50))            # after 50 steps, first movement 1.0
print(cert.steps_for(change=0.5, tolerance=1e-3))   # warm-start budget after a drive change
print(cert.to_dict())                               # fields, rate, verdict and limit
```

| Name | Meaning |
| --- | --- |
| `row_mass`, `lipschitz`, `dt` | `rho`, `L` and the model's integration step |
| `adaptation`, `certified` | whether the model adapts; `certified` is `rate < 1` without adaptation |
| `rate`, `mass_limit` | the factor `q`; `1 / L`, the row mass below which the model is certified |
| `error_bound(movement)` | `movement / (1 - q)` for each entry; `inf` when uncertified |
| `apriori_bound(first_movement, steps)` | `first_movement q^steps / (1 - q)`; `inf` when uncertified |
| `steps_for(change, tolerance)` | warm-start steps; 0 for no change, at most 1 at rate 0; `ValueError` when uncertified |

`error_bound` takes potential movement. `NeuralGraph.residual` reports the equation error
before multiplication by the step, so pass `brain.neuron_model.dt * brain.residual(drive,
state)` (unmasked free phase). The potential bound times `cert.lipschitz` bounds activation
error. `settle_batch(tolerance=...)` measures activation movement and must not be substituted.
The warm-start budget assumes the old state is the old equilibrium; an approximate warm state
adds its own remaining error.

The certificate covers the free phase without adaptation. A nudge adds a drive the argument
omits, adaptation adds a slow variable, and records are read and written without settling. For
those, and above the limit, use `residual` and `equilibrate`; they qualify the equations at the
measured state and say nothing about uniqueness or convergence. A failed certificate means only
that this sufficient argument does not apply.

### Using it

- Report `cert.to_dict()` in every receipt next to the residual.
- Recompute `certificate(brain)` after learning.
- Keep learned brains under the limit where you can: bound efficacies, scale the gain by the
  fan-in and watch `row_mass` over a life. `LearnerConfig.scale_cap` (default 8) clips each
  plastic efficacy separately, which does not bound a row sum.
- On warm-start streams the step bound is logarithmic in the change. A certified brain may
  need more steps than a given finite budget allows.
- Lesions keep the certificate: ablation only removes synapses and lowers the row mass.

### Equilibrium-propagation scope

`cd.ep_structure(brain, *, fixed_inputs=(), tolerance=1e-12)` returns an `EPStructure`, a
structural check that the graph admits the symmetric free-state argument of
[equilibrium propagation](https://arxiv.org/abs/1602.05179). It is separate from the
contraction certificate. Repeated or out-of-range `fixed_inputs` and a negative or nonfinite
`tolerance` raise `ValueError`. An apical graph also raises `ValueError`, because its
state-dependent normalization falls outside the argument whatever its reciprocal symmetry. Its
held-gain rule is a local contrast without this gradient theorem; the
[depth benchmark](../benchmarks/depth/README.md) measures its usefulness.

| Field | Meaning |
| --- | --- |
| `fixed_inputs`, `free_neurons` | the declared inputs and the count of the other neurons |
| `free_asymmetry` | largest gap between a free/free effective weight and its reverse (parallel edges summed) |
| `fixed_incoming_mass` | largest incoming effective mass into a declared input; nonzero means it is no source |
| `adaptation`, `tolerance` | adaptation present; the tolerance of both checks |
| `compatible` | both gaps within `tolerance` and no adaptation; `to_dict()` includes it |

Source inputs settle to potentials set by their fixed drive and bias, and their projections act
as external fields on the free neurons without reverse edges. Keep them unchanged and unnudged
between phases. Efficacy tying alone does not give effective symmetry when contact counts or
gains differ. For the contrast to be a gradient, phases must also converge on a smooth stable
branch, and finite beta keeps an estimation bias. Convert the raw contrast as in
[learning](#5-why-the-contrast-is-a-gradient);
[`tests/test_equilibrium.py`](../tests/test_equilibrium.py) checks it against finite differences
and [`tests/test_runtime_checks.py`](../tests/test_runtime_checks.py) the refusals.

### Related results and their assumptions

Keep five questions separate: whether an equilibrium exists, whether it is unique, whether
this iteration reaches it, whether the continuing life preserves useful context, and whether
its behaviour succeeds. Neither the certificate nor residual qualification answers the last
two. Agreement means satisfying the `F_c` relations; neurons need not share an activity. Check
each result below against the executed update before reusing it.

[Dijkstra (1974)](https://www.cs.utexas.edu/~EWD/transcriptions/EWD04xx/EWD426.html) defines
self-stabilization: legitimate states, moves that preserve them, and specific local algorithms
reaching them from every start under a central daemon. A recovery test reuses the pattern by
naming the disturbance, the recovery condition and the continuation it must survive. The
Cadence life has no arbitrary-state self-stabilization guarantee.

[Bertsekas (1983), sections 3 to 4.2](https://web.mit.edu/dimitrib/www/Distr_Comp_Fixed.pdf)
proves asynchronous convergence for a map contracting in a weighted maximum norm, if every
coordinate keeps updating and stale reads eventually expire. The row-mass certificate is such a
maximum-norm contraction for the fixed free map; Cadence settles synchronously.

[Simonetto (2017), Theorem 3.1(b)](https://arxiv.org/abs/1704.07338) tracks a moving fixed
point: for maps contracting by `q` on a common domain and norm, with iterate error `eta` and
drift `delta` per step, `e[t] <= q^t e[0] + (eta + delta) (1 - q^t) / (1 - q)`. Observation,
trace, recalled records and weights all move Cadence's target.
[Bernstein and Dall'Anese (2018)](https://arxiv.org/abs/1804.09768) add delays and drops.

[Hansen and Ghrist (2019)](https://arxiv.org/abs/1808.01513) define the sheaf Laplacian
`L = D* D` for fixed restriction maps; its kernel holds the compatible assignments, whose local
values may differ. [Their 2021 paper](https://arxiv.org/abs/2005.12798) shows that diffusion
`dx/dt = -alpha L x` converges to the projection onto `ker L`, and a held boundary gives the
nearest harmonic extension. It is a symmetric comparison model, distinct from Cadence's directed
nonlinear operator.

[Hanks et al. (2025)](https://arxiv.org/abs/2504.02049v2) extend the flow to strongly convex
edge potentials, `dx/dt = -alpha D* grad U(D x)`, and
[Zhao et al. (2025)](https://arxiv.org/abs/2510.00270v1) to asynchronous updates with bounded
stale reads and a small step. Cadence's activation, directed weights, nudges and memories lack
this gradient form; the gradient of a squared residual adds a Jacobian transpose.
