# Learning: the free/nudged rule

Start with [one continuing equilibrium brain](world-model.md) and
[the continuing-brain quickstart](quickstart.md#observe-act-and-learn).
This page explains local prediction repair and demonstrations without adaptation: the neuron
equations, a numerical update, the gradient assumptions, and configuration choices.
This rule changes the synapses of the settled regions. A [records cortex](memory.md#records)
learns consequences and reward with one read and one delta-rule write per outcome, without
settling phases.

The isolated classification, calibration and derivative examples here are
mechanism controls. They help diagnose local plasticity; they are not a substitute
for a continuing brain whose memory and acquired relations survive use and
correction. Keep the teaching signal absent when measuring acquired answers.

For continuous action-return targets on a jointly settled recursive graph, use
the [recursive-training guide](recursive-training.md). It uses `PatchNet.observe`
to gate updates on full-equation checks, and optionally local energy curvature,
rather than treating a fixed number of relaxation steps as a settled phase.

## 1. What learning changes

A `NeuralGraph` carries three parameter arrays:

- `efficacy`, one signed synaptic efficacy per synapse. The effective drive of synapse `e`
  per unit of presynaptic activation is `gain · count[e] · efficacy[e] · exp(log_gain[pre[e]])`;
  in a brain built by `layered` with `learning_neuron_model()`, `gain`, `count`, and
  `log_gain` are 1, 1, and 0, so the drive is just `efficacy[e]`. Call it `W[e]`.
- `bias`, one number per neuron.
- `log_gain`, one per neuron, which the learner leaves alone.

Learning moves `efficacy` and `bias`. The connectome stays fixed: which neurons synapse
onto which never changes, and learning changes only how strongly each synapse drives.

## 2. What settling computes

Each neuron `i` holds a potential `v[i]` and publishes an activation `s[i]`. One step of the
neuron model, applied to every neuron at once:

    synaptic_input[i] = sum over synapses e with post[e] = i of  W[e] · s[pre[e]]
    total[i] = synaptic_input[i] + stimulus[i] + bias[i]     (+ nudge[i] in a nudged phase)
    v[i]    <- v[i] + dt · (total[i] - v[i])
    s[i]     = act(v[i])

Each neuron relaxes its potential toward its total input. `act` is the neuron model's
activation function. `learning_neuron_model()` sets slope 1, threshold 0, and leak 0.1:

    act(v) = tanh(v / 2)          for v >= 0
    act(v) = 0.1 · tanh(v / 2)    for v <  0

so rest (`v = 0`) publishes exactly 0, positive drive saturates toward 1 around `v ≈ 4`,
and negative drive publishes a small negative number instead of a hard zero. Settling
repeats this step from rest (or from a given state) until no neuron's activation moves
more than `tolerance` in one step, or until the step cap. A readout sees that state.

An input enters as a *stimulus*, a fixed drive on the input neurons with one number per
neuron (a pixel's level, a board cell's occupancy). In a `layered` connectome, input
neurons have no inbound synapses. With zero input bias, at equilibrium each sits at
`v = stimulus` and publishes `act(stimulus)`. Other connectomes can feed back into inputs.

## 3. The two phases and the update

For a batch of inputs with targets:

**Free phase.** Settle under the input stimulus alone. Call the rest state `s⁰`. The output
neurons' activations are the brain's answer; no target has entered.

**Nudged phases.** From `s⁰`, settle again with an extra drive on the output neurons only:

    nudge[i] = beta · weight · (target[i] - p[i])      for output neurons i
    p        = softmax(s[outputs] / T)

`p` is a softmax over the output neurons, so pushing the target's neuron up pushes the
others down by their share (a cross-entropy nudge). `weight` is 1 for a label; for a
reward it is the advantage of the action (see [reward](reward.md)). With `centered=True`
(the default) two nudged phases run from the same `s⁰`: one with `+beta`, rest state
`s⁺`, and one with `-beta`, rest state `s⁻`. The nudge travels back over the feedback
synapses, so hidden neurons rest at slightly different activations in the two phases.

**Update.** Every synapse reads its own presynaptic and postsynaptic neurons in the two
phases, and every neuron reads itself:

    contrast[e]  = mean over the batch of ( s⁺[pre[e]] · s⁺[post[e]] - s⁻[pre[e]] · s⁻[post[e]] ) / (2 beta)
    bias_term[i] = mean over the batch of ( s⁺[i] - s⁻[i] ) / (2 beta)

    efficacy[e] += eta   · contrast[e]
    bias[i]     += eta_b · bias_term[i]

The synapse term is a local, Hebbian-like quantity: the product of presynaptic and
postsynaptic activity in one nudged phase minus the same product in the other.

Two more details, both in `Learner.update`. With `reciprocal=True` (the default), a synapse
and its reverse (`i → j` and `j → i`) form a *reciprocal synapse pair* with one shared
efficacy, so their two contrasts are averaged and both directions move by the same amount.
A plastic efficacy's magnitude is clipped to eight; frozen efficacies are preserved.
With `centered=False` the second nudged phase is skipped and `s⁻` is replaced by `s⁰` with
`beta` in place of `2 beta`.

The implementation retains the free and nudged states and optimizer history;
it does not retain a backward computation graph. Softmax reads its declared output
group, and weight tying aggregates the members of a declared tie group. These two
operations read a group of neurons or synapses, beyond the per-synapse contrast.

## 4. One observation repairs a settled prediction

For a custom world model, save a free prediction under the current observation
and proposed action with `Learner.free`. Only after the environment executes the
action does its observed consequence enter the teaching target.
Both nudged phases start from that same free state under the original drive.
Their endpoint contrast changes synapses for future encounters.

The records cortex learns consequences with one read and one write. Use this settled
repair where the prediction must complete a partial reading, or where the readout is the
policy under late credit. The repair assigns no credit across delays: use explicit
temporal state and [reward eligibility](reward.md) where the task needs them. Score
outcomes before updating, and keep imagined or teacher-nudged states out of
witnessed-event memory.

## 5. Why the contrast is a gradient

Let `F` be the recurrent neurons and hold the input activations fixed. With no adaptation,
symmetric **effective** recurrent weights, and a monotone differentiable activation,
the continuous-time dynamics descend

    E(v_F) = Σ_{i∈F} ∫₀^{v[i]} u · act'(u) du
             − ½ Σ_{i,j∈F} W[i→j] s[i] s[j] − Σ_{i∈F} d[i] s[i]
    d[i] = stimulus[i] + bias[i] + Σ_{k input} W[k→i] s[k]

The one-way input projections belong in `d`, outside the symmetric recurrent sum.
A finite Euler step need not lower this energy. A stationary point need not be a
minimum or unique. Tying `efficacy` alone does not make effective weights symmetric
if opposite contact counts or presynaptic gains differ.

On a smooth stable equilibrium branch, converged free/nudged phases obey the
[equilibrium-propagation identity](https://arxiv.org/abs/1602.05179): the limiting
contrast is minus the loss gradient with respect to the weight of a reciprocal synapse
pair. The pair has one weight, so its two identical directed contrasts
must not be summed twice. For a quadratic nudge the loss is half squared error.
Cadence's cross-entropy drive omits `1/T`, so its loss is **T times cross-entropy**.
The [centered estimator](https://arxiv.org/abs/2006.03824) cancels the leading nudge
bias on a sufficiently smooth branch; a finite nudge crossing a kink or a different
attractor need not have that accuracy.

At a finite nudge the contrast is an estimate of the derivative, with two error terms. On a
branch whose third derivative is bounded, the centered bias is of order `beta^2`. A phase
solved only to a sup-norm error `eps` adds at most `A * (eps_plus + eps_minus) / beta`, with
`A` a bound on the activities, because each endpoint product moves by at most `2 A eps` and
the contrast divides by `2 beta`; a smaller nudge therefore needs a tighter solve. The
free-phase contraction certificate does not cover the nudged phases by itself: a nudged
phase is a contraction when `L * (rho_W + beta * K_C) < 1`, with `K_C` the Lipschitz constant
of the nudge, which is `1 / (2 T)` for a normalized softmax slot under `T` times
cross-entropy. The default learning activation has a kink at zero, so the smooth-branch
argument excludes equilibria at or across it; `leak=1` with `slope=1` gives the smooth
`tanh(v/2)`. The rule is measured directly in
[benchmarks/local_equilibrium](../benchmarks/local_equilibrium/README.md) and its longer
follow-up: on a nonlinear agreement judgment the public centered update matches an
implicit-gradient learner on the same network, after a first 800-update run that missed its
robustness criterion; both arms rescale the free/free rows to a row-mass cap after each
update, a global projection that keeps the contraction budget and is not a local operation.

`Learner.contrast` returns a statistic, and `Learner.update` keeps the library's
step convention. When `c[e] = gain * count[e] * exp(log_gain[pre[e]])`
is not one, conversion to minus the gradient with respect to a tied `efficacy`
requires multiplying by `c[e]` (assuming the factors of the two directions agree). For
ordinary cross-entropy also divide by `T`. Bias gradients need the same loss scaling but no
contact factor. Sharing a parameter across several distinct synapses gives a sum of
contributions; the implementation uses their mean as a step-size convention.
Explicit tie groups and reciprocal pairs are combined transitively. Tying constrains
increments, so tied initial efficacies must agree if their values should remain equal.
Frozen members contribute zero to the averaged increment and remain fixed.
`tests/test_equilibrium.py` checks the scale conversion against finite differences.

A small activation change is not a fixed-point certificate: saturation or tiny `dt`
can make it small while the potential is far from rest. Check the neuron equations:

```python
import numpy as np
import cadence as cd

connectome = cd.layered(4, 16, 2, seed=0)
learner = cd.Learner(cd.NeuralGraph(connectome, cd.learning_neuron_model()), connectome.populations["output"])
drive = np.random.default_rng(0).random((8, connectome.n))
free = learner.free(drive)
remaining = learner.brain.residual(drive, free)  # one diagnostic value per batch row
```

For a nudged phase, pass its actual `nudge` too, and inspect the result before changing
the brain's parameters. The diagnostic uses one transport evaluation and does not settle
again. `learning_neuron_model` provides responsive defaults, but cannot guarantee convergence,
uniqueness, or accurate credit for every connectome. The leak keeps a small response
below rest; it does not remove saturation or make the piecewise activation globally smooth.

Select `qualified=True` to require qualified teaching. The same local
contrast then runs only after the free and required nudged phases meet the full
potential and adaptation residual. `damping` permits a bounded number of
integration-step halvings within each phase's existing sweep budget; every
returned phase is checked against the original model's equations. These
numerical choices are candidates to compare with finite teaching on acquired
free behavior.

This component example supplies a fresh `LearnerConfig`, including its
`eta=0.2` and `momentum=0.0` defaults. To change qualification while preserving
the composition's learning rates and momentum, use the
[named-override recipe](brain.md#settle-and-check).

```python
brain = cd.Brain.compose(
    inputs=4, actions=2, modules=(8,), seed=7,
    learning=cd.LearnerConfig(
        qualified=True, damping=3, free_steps=512, nudged_steps=512,
        tolerance=1e-7,
    ),
)
observation = np.array([[1.0, 0.0, 0.0, 0.0]])
taught, report = brain.learner.step(
    brain.stimulus(observation, memory=False), np.array([0]),
)
answer = brain.predict(observation)  # free qualified recall; inspect whether it is correct
```

`LearningPhaseError` exposes the failed phase, completed phase diagnostics and
attempted work. A refusal makes no parameter or optimizer write. Successful
phase qualification certifies the equations at those states; the gradient
interpretation still needs the symmetry, branch, contact and loss assumptions
above. A qualified lesson alone does not establish useful acquisition or
retention. Direct `contrast` and `update` remain lower-level operations on
supplied states and cannot certify their origin.
The `ActorCritic` reward-eligibility path retains its finite nudged phase;
selecting qualified supervised learning does not certify that separate path.
Its `ActorCriticConfig.eligibility_steps` budget is independent of supervised
nudges; the built-in Brain reward configuration uses 12, while `None` in an
explicit reward configuration inherits the learner's nudged budget.

### Qualified teaching budgets

`qualified=True` changes what `free_steps` and `nudged_steps` mean. Finite
teaching reads the contrast after at most `nudged_steps` sweeps, so a small
nudged budget is a teaching-phase length. Qualified teaching checks every phase
against the full equations within its budget, so both numbers are settle
budgets: a nudged phase typically needs about as many sweeps as the free phase
it starts from, because the nudge moves the equilibrium. A qualified
configuration whose nudged budget is far below its free budget refuses lessons
the free budget would settle; at `Brain.compose`'s teaching defaults
(`free_steps=1024, nudged_steps=12`) it refuses every lesson on realistic
input, with the data playing no part.

Constructing a `LearnerConfig` with `qualified=True` and
`nudged_steps < free_steps` therefore warns, including through
named overrides or `dataclasses.replace`, and a refused nudged or
opposite phase under such a configuration names the budget mismatch in its
`LearningPhaseError` message and `hint`. Set `nudged_steps` comparable to
`free_steps` when opting in, as in the
[named-override recipe](brain.md#settle-and-check); a deliberately
small settle budget remains allowed, and the warning can be filtered. The
finite default and the reward-eligibility contract above are unchanged.

### Finite free phases that run out of budget

Finite teaching reads the contrast from whatever state its budgets reached: "a
brain that does not settle refuses to act" holds for answers, not for finite
lessons. A free phase that uses its entire `free_steps` budget usually never
reached the movement stop, so the lesson was learned from a state that may not
have settled; on dense real input this can silently consume a whole run
(issue 127). `Learner.step` therefore counts it as `free_budget_exhausted` in
the lesson report (`demonstration_free_budget_exhausted` through `Brain.step`)
and warns, with the equation residual available as `free_residual` in the same
report. Phases with `tolerance=None` are declared fixed-length and stay
silent. The update law is unchanged: raise `free_steps`, lower the gain, or
opt into `qualified=True` to refuse such lessons instead.

Lesson reports distinguish attempted and accepted row presentations. They count
every required free/positive/negative sweep, residual transport and complete-state
stagnation comparison, with row-weighted totals for batches. `Brain.step` keeps
the same fields in `last_learning` with a `demonstration_` prefix, including work
from a refused lesson. Previously accepted real feedback stays recorded there.

### Acquisition, working memory and durable retention

Learning 24 cue/action relations does not require holding 24 cues in working
memory at once. Acquisition asks whether teaching changes a later free answer.
Working-memory recall asks whether a vanished cue remains available across a
delay or distractors. Durable retention asks whether a previously learned
relation can be recalled after other learning, with temporary state cleared.
Measure these separately, and declare the finite storage and teaching exposure.

`Brain.predict` and `accuracy` exclude both the working trace and associative
store. They test the graph's learned parameters. `act` also reads those memory
ports, so an answer can depend on consolidated associations. For a durable-store
test, clear live state with `brain.reset()` and reset the associative store's
fast residual with `brain.hippocampus.reset(batch)` before each independent
query; its consolidated weights remain. `Brain.reset` alone keeps hippocampal
records. A passed store-assisted answer does not show that the graph's contrast
rule acquired the relation.

Teacher demonstrations in `step` teach the slow graph. Actual rewarded actions
in `learn` write the associative store; they are different observed targets.
Do not invent rewards to populate memory. Rehearsal can help preserve learned
relations, but every repeated observation and learning phase counts as experience
and work. Compare it with the declared no-rehearsal control. Use unambiguous
cue/target pairs: conflicting labels on an identical independent observation
cannot certify a deterministic fresh-state mapping without more context.

## Feedback and the reach of a nudge

A contrast is zero if neither the presynaptic nor the postsynaptic neuron changes under
the nudge. Hidden neurons therefore need a directed feedback path from nudged outputs.
`layered` supplies tied feedback between hidden and output neurons. A general
directed graph may provide some such paths and omit others. Asymmetry among free neurons
breaks the energy-gradient argument above. One-way projections from fixed external inputs
do not: a source neuron at its unchanged equilibrium supplies a constant field to the free
subsystem. `ep_structure(brain, fixed_inputs=...)` checks this structural distinction; it
does not check convergence or certify finite-beta accuracy. Feedback reachability alone does not ensure
useful credit if activations saturate or phases fail to converge.

For a measured connectome, distinguish the supplied topology from an assumed learning
mechanism. Adding reverse synapses changes the model. Test that modelling choice
with controls rather than treating it as a biological consequence.

## 6. Using it

For an individual prediction head, construct `(batch, connectome.n)` drives from the
pre-action context. `step` takes integer outcome/action indices within the output group;
for `slots`, use one index per row and slot. `accuracy` is the fraction of correct
choices over all rows and slots. Use [explicit target patterns](tasks.md#pattern-targets)
for regression or reconstruction.

Learning rates such as `eta` and `eta_bias` are hyperparameters: they configure
the local update, while efficacies and biases are learned parameters. Choose
rates on development tasks and freeze them before confirmation. A rate that
helps one acquisition task is not a universal default. Across-brain selection
may treat these hyperparameters as genes; that does not make them learned
within a life.

For a composed brain, use `brain.retune(learning_eta=..., learning_eta_bias=...)`
to change teaching rates while preserving its other settings. Reward learning
has independent `actor_*` rates; see the [ownership table](brain.md#defaults-and-expert-overrides).

For a standalone `Learner`, `LearnerConfig` is frozen. To change the learning
rate between completed updates:

```python
from dataclasses import replace

# After a completed update on your own Learner:
learner.config = replace(learner.config, eta=0.5)
```

`learner.brain` is a plain `NeuralGraph` at every moment. Settle it, run `conformance` on
it, export `learner.brain.dense()` for a page, or put its `to_dict()` in a receipt.

### Calibrating the operating point

`learner.calibrate(drive, level=None, grid=None)` chooses a global synaptic
gain by its free output operating point. Left unset, the target is
competitive: the mean over rows and output slots of the highest output
activation, at 0.5 — one choice up, the others wherever the competition puts
them. An explicit `level` instead targets the mean over every output neuron.
On a readout of n choices that mean target pushes all n up together: driving
36 word outputs to a mean of 0.5 selected gain 12 and later 128 and saturated
the readout before the winner was useful (issue 125), which is why the mean
target is no longer the default. Either way this is an operating-point
heuristic: the sampled gains may never reach the target, and reaching it does
not establish useful credit, acquisition or retention. Use training or separate
calibration inputs, then freeze the chosen gain before final evaluation.

The default grid tries the current gain first, followed by its multiples `2**k` for
`k=-8, …, 8` excluding zero. It spans from `gain / 256` to `gain * 256`
where those candidates are representable; exact ties keep the current gain.
An explicit positive, finite `grid` keeps its supplied values and order without
additional candidates.

Every candidate starts from rest with the same input batch and `free_steps`
budget. With `qualified=True`, calibration uses `tolerance` and bounded
`damping` to check the candidate's full equations, and excludes a failed or
nonfinite state. Finite mode retains the finite settling contract and reports
that qualification is not required. If no candidate is usable, calibration
raises `RuntimeError` and preserves the original graph and optimizer history.
A successful call installs the tested winner and returns its gain; it does not
perform a teaching update.

Inspect `learner.last_calibration` for the candidate gains, gaps, residuals,
qualification and total attempted solve work, including refused candidates.
Its counters include sweeps, residual transports, stagnation comparisons and
activation-cache checks; row counters multiply each operation by the batch size.
Charge this work separately from lessons and free queries. A missing gap marks
a refused candidate, not an observed activity at the requested level.

Global gain scales all synaptic drive, including inhibitory connections.
`calibrate_bias` instead searches bias values for declared populations or
neurons. Its default coordinate bisection uses finite settling and a bounded
span; it can return an endpoint or miss a target when populations interact.
Neither this helper nor the finite `preflight` diagnostics inherit a learner's
qualification setting. An empty preflight warning list is not an equilibrium
certificate.

`calibrate_bias(..., qualified=True, damping=3, report=...)`
requires each midpoint and the final candidate to meet the original equations
for every input row. Every solve has its own `steps` budget; the report retains
all attempted work, final observed means and target gaps. A refused solve
raises `RuntimeError` and leaves the supplied graph unchanged. Even a qualified
candidate can miss its target: bisection still needs a suitable response over
its chosen span. Inspect the observed means before installing the bias array.

Finite bias searches also validate every midpoint
and the final state, even without a report. Nonfinite potentials, activation or
adaptation, or inconsistent activation, raise `RuntimeError` before their
output can steer the search. The source graph remains unchanged. Finite states
with a large equation residual remain permitted; qualification is still opt-in.

This small example calibrates a motor population on training drives before
any teaching. It removes motor lateral connections explicitly while keeping
the processing/motor feedback, checks the result, then installs it:

```python
operating = cd.Brain.compose(4, 3, modules=(8,), lateral=0.0, seed=9)
training_inputs = np.array([[1.0, 0.2, 0.0, 0.1], [0.2, 1.0, 0.1, 0.0]])
training_drive = operating.stimulus(training_inputs, memory=False)
calibration_report = {}
initial_graph = operating.brain
candidate_bias = cd.calibrate_bias(
    initial_graph, training_drive, {"motor": 0.25},
    qualified=True, damping=3, steps=256, tolerance=1e-7,
    report=calibration_report,
)
candidate_graph = initial_graph.with_parameters(bias=candidate_bias)
checked = candidate_graph.equilibrate(
    training_drive, budget=256, tolerance=1e-7, damping=3,
)
assert np.all(checked.qualified)
motor_mean = checked.state.activation[:, operating.motor_index].mean()
assert abs(motor_mean - 0.25) < 0.01
operating.learner.brain = candidate_graph
print("motor mean", round(float(motor_mean), 4), "residual", checked.residual.max())
print("target gaps", calibration_report["target_gaps"],
      "calibration row sweeps", calibration_report["total_row_sweeps"])
```

The target, lateral weight and gain are initialization choices, not learned
facts about a biological brain. Keep hand-set values as controls when selecting
these candidate genes. Count calibration and the independent final check
separately from teaching, and test acquired free behavior afterward.

## 7. Every knob

`LearnerConfig()` has standalone defaults. `Brain.compose(learning=None)` uses
the [composed founders](brain.md#defaults-and-expert-overrides); that table is
also the guide to each `learning_*` and `actor_*` setting's operation.

Passing `learning=LearnerConfig(...)` uses that configuration; it does not merge
its defaults with the implicit `Brain.compose` settings. Both contexts use
finite teaching by default. The `qualified` and `damping` options below
select full-equation teaching and its numerical strategy. Rates and temperature can be selected
on development data and frozen before confirmation; recommendations are not
constructor defaults. `eta_bias` left unset derives `eta / 10` at construction;
once resolved it is an ordinary explicit value. Both named overrides and
`dataclasses.replace` keep the old bias rate unless it is set too or passed as
`None` to re-derive it. A bias rate above the synapse rate warns:
at `eta=0.0015` the former fixed default of 0.02 made the bias step thirteen
times the synapse step (issue 126).

| knob | where | what it does | standalone default / selection notes |
|---|---|---|---|
| `beta` | `LearnerConfig` | nudge strength; smaller is closer to the gradient, larger a stronger signal | 0.1 |
| `free_steps`, `nudged_steps` | `LearnerConfig` | the most steps a free and a nudged phase may take before the contrast is read | 100, 50; under `qualified=True` both are [settle budgets](#qualified-teaching-budgets): keep them comparable |
| `tolerance` | `LearnerConfig` | activation-movement stopping for finite phases; full-equation residual when qualified (`None` only for finite phases) | 1e-4 |
| `qualified`, `damping` | `LearnerConfig` | require full-equation qualification before teaching; allow bounded integration-step halvings within each phase budget | `False`, 3; compare candidates with the finite control; qualification needs a [nudged budget comparable to the free one](#qualified-teaching-budgets) |
| `eta` | `LearnerConfig` | efficacy step; the contrast is divided by `2 beta` | default 0.2; select on development data |
| `eta_bias` | `LearnerConfig` | bias step | unset derives `eta / 10` (0.02 at the standalone `eta`); explicit values are kept, and a bias rate above `eta` warns |
| `temperature` | `LearnerConfig` | softmax temperature of the cross-entropy nudge; also the policy temperature when sampling actions | default 0.2; select any alternative on development data |
| `centered` | `LearnerConfig` | contrast `+beta` against `−beta` (two nudged phases) rather than against the free state | `True` |
| `nudge` | `LearnerConfig` | `"cross_entropy"` or `"quadratic"` (`beta · (target − s)`) | cross-entropy for classes |
| `momentum` | `LearnerConfig` | each efficacy steps on a bias-corrected running average of its contrast | 0 (off); tune with the learning rate |
| `decay` | `LearnerConfig` | every update shrinks each plastic efficacy and bias by this fraction | 0 by default; decay also forgets useful weights |
| `homeostasis_rate`, `homeostasis_target` | `LearnerConfig` | at every teaching or reward update each output neuron's bias moves by the rate times the shortfall of its free activation below the target, negative when above: the readout's intrinsic plasticity, which keeps it out of saturation (no slope for a nudge) and out of silence (every contrast tiny) | 0 (off), 0.3; both are genes; a rate on the order of the lesson rate, as the recall chamber's development receipts measure |
| `normalize`, `normalize_floor` | `LearnerConfig` | divide each efficacy's step by its bias-corrected running RMS plus a floor | 0 (off), `1e-3`; select on development data; combining momentum and RMS gives an Adam-style update |
| `reciprocal` | `Learner` | tie each synapse and its reverse into a reciprocal pair with one efficacy | `True` |
| `plastic_synapses` | `Learner` | bool per synapse; the others keep their efficacy | all |
| `plastic_neurons` | `Learner` | bool per neuron; the others keep their bias | all |
| `synapse_rate` | `Learner` | a nonnegative step multiplier per synapse, applied before tying | `None` (every synapse at `eta`) |
| `leak`, `slope`, `dt` | `learning_neuron_model` | sub-rest response, activation slope, integration step of the neuron update | 0.1, 1.0, 0.5; `Brain.compose` uses `dt=1.0` |
| `density`, `feedback`, `lateral`, `init`, `skip` | `layered` | input→hidden density, feedback scale, output↔output scale, initial magnitude, direct input→output synapses | 0.3 (the default; 1.0 for small brains), 1.0, 0, 1.0, `False` |

`Learner.parameters()` counts plastic efficacies (a reciprocal pair or tie group counts
once) plus plastic neuron biases. A plastic synapse whose reverse is frozen still contributes
one trainable efficacy. Report optimizer arrays and episodic memory separately when comparing
storage; parameter count alone does not measure execution cost.

## 8. Warm starts, costs, and what to expect

A free phase may start from an earlier state (`learner.free(drive, warm=state)` or
`learner.step(..., warm=state)`). Within the same attracting basin this can save settling
steps; with multiple attractors it can change the answer. A capped warm phase deliberately
retains transients and is not necessarily an equilibrium. Reset state at independent
episode boundaries and compare warm and cold settling on changing inputs.

Each centered update runs three settling phases. Measure all phase steps and wall time; fewer
epochs do not by themselves mean greater sample efficiency or lower compute. When the
task is to learn what follows a reading, a [records cortex](memory.md#records) reads with
one product and writes with one delta-rule step per outcome, without settling phases.

### The admitted step, and a life lived online

The patches of the [belief](belief.md) family learn by a step that is admitted, not taken: the
chunk is replayed from its boundary under the proposed parameters and the largest halving that
lowers the loss by the Armijo margin is taken, starting from twice the last admitted step and at
most `rate`. Every lane that learned by the plain step diverged or learned nothing, so the
admission is the default and the plain step is the control (`backtrack=False`). Two regimes
follow from the rung demos. By day, batches of whole streams from a fresh boundary at a high
cap (the night nursery: eight streams of 64 moments at a cap of 4, 120 epochs, the best
checkpoint kept by held-out error; short days of 24 to 40 epochs leave a steering patch's
weighing on the wrong sense). By night, one stream lived online in chunks of eight moments at a
cap of 0.1, the boundary carried, the cortex asleep while the steering patch learns
(`Steered.run(learn_cortex=False)`) or gated by a governor (`Life`); the day's cap on the
night's chunks wrecks tracking, and a steering patch's rate scale above one pins a sense.
[How to build a rung demo](howto-rung.md) has the numbers.

## Rates under normalization

With `normalize > 0`, each plastic efficacy and neuron bias has a running
second moment of its own raw update signal. Before masks, synapse-rate
multipliers, tying, decay and efficacy clipping, the proposed increment is:

```text
increment = rate * signal / (sqrt(bias_corrected_second_moment) + floor)
```

`rate` is `eta` for efficacies and `eta_bias` for neuron biases. `signal` is the
raw contrast, or its bias-corrected running mean when `momentum > 0`; the second
moment always uses the raw contrast. The learner's floor is `normalize_floor`.
For a first update with raw contrast `g`, the formula reduces to
`rate * g / (abs(g) + floor)`. A consistent signal much larger than the floor
therefore gives increments close to the rate in magnitude. Quiet signals are
attenuated, a zero signal with no momentum history stays zero, and changing
signals or momentum history can give different sizes, including sizes above
the rate. These are parameter increments, not fixed changes in effective
synaptic drive or a promise that every synapse moves.

The composed teaching default `eta=0.5` and actor default `eta=1.0` use
`normalize=0`. Enabling RMS normalization can make their updates much larger
relative to a small raw signal and can saturate outputs. Retune both `eta` and
`eta_bias`; a resolved configuration keeps its bias rate when `eta` changes.
For a composed brain, set paired `learning_*` or `actor_*` rates with `retune`,
or pass the respective bias override as `None` to re-derive `eta / 10`. A starting development sweep
of `eta=0.001` to `0.003` is motivated by the reported pilots below, but is not a
universal safe range or a replacement for task measurements. Select the bias
rate independently and check acquisition, retention and actual update sizes.

Both `LearnerConfig` and `ActorCriticConfig` emit `RuntimeWarning` at construction
when `normalize > 0` and either `eta` or `eta_bias` exceeds `0.05`. This is a
conservative diagnostic threshold, not a stability bound: smaller rates can
also fail. The warning does not change rates, optimizer equations or defaults.

A normalized composed default was proposed from the pilots above and measured
against the capability suite on 2026-10-04 (issue 131): with
`eta=0.003, normalize=0.99, momentum=0.9` for both composed learners,
supervised acquisition contracts still passed, but the reward stream failed to
re-adapt after a contingency change (0.486 against the 0.9 adaptation
contract, with the pre-change bootstrap passing) — absolute per-synapse steps
at that scale cannot overturn an established policy quickly. The composed
defaults therefore remain unnormalized; normalized rates stay per-application
settings selected on development data, with the construction warning and this
section as the guardrails.
The actor's normalization and critic's separate rate are described in
[the reward guide](reward.md#rates-under-normalization).

[Issue 131](https://github.com/muellerberndt/cadence/issues/131) reports the
application observations that motivated the warning:

| Reported pilot | Observation and scope |
| --- | --- |
| Atari, Cadence 0.70.0 | Actor `eta=0.003` rose and collapsed to a held action; `0.001` was reported as more stable. The [qualification](https://github.com/muellerberndt/cadence/issues/97#issuecomment-5947900523) established held-action retention and continuation, with no measured improvement from reward. |
| [Transcribe, Cadence 0.71.0](https://github.com/muellerberndt/cadence-transcribe/blob/26d8667a33eec38df35bddd9e743b32c0d28dd23/STATUS.md) | At 20,000 rows, P03 (`eta=0.03`, `eta_bias=0.003`) scored 0.048 top-1; P01 (`eta=0.003`, `eta_bias=0.02`) scored 0.441. Both rates changed, so this comparison does not isolate the effect of `eta`. Other architectures with `eta=0.003` reached 0.53. |
| [Patch World v2, Cadence 0.72.0](https://github.com/FloatingPragma/oph-meta/blob/fd36f5b58ca7b3efcbd99db0d0490be4bae5de67/cadence-patchworld-v2/STATUS.md) | Actor `eta=0.2` was reported to produce stereotyped actions; `0.002` reached 0.53 eat-when-hungry within 500 ticks in a lone-creature assay, versus 0.20 for uniform random actions. Random actions still outlived and outbred the learning brains in the reported population assays; the eating result does not establish learned navigation. |

These reports motivate checking normalized rates. They do not constitute a
matched cross-task rate study or establish an improved learning algorithm in
0.72.1. Transcribe's [pilot configurations](https://github.com/muellerberndt/cadence-transcribe/tree/26d8667a33eec38df35bddd9e743b32c0d28dd23/protocols/pilot071)
and [base learning settings](https://github.com/muellerberndt/cadence-transcribe/blob/26d8667a33eec38df35bddd9e743b32c0d28dd23/transcribe/brain.py)
retain the independent efficacy and bias rates.
