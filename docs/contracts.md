# Numerical and learning contracts

Each model family in Cadence solves its own equations, stops by its own rule and changes
its parameters under its own contract. This page sets them side by side. Read the row for
the family you use before reading a result off it, and before changing a solver or a
learning rule.

## Choose the computation

| API | Equations | Stopping rule | Update contract |
| --- | --- | --- | --- |
| `NeuralGraph`, `Learner` | Rate neurons exchange activity over a directed graph ([concepts](settling.md#settling-and-equilibrium)). | `equilibrate` checks the full fixed-point equations; `settle` alone may run a fixed budget or stop on activity movement. | Free/nudged local contrast, finite phases by default; optional reward traces. The gradient reading needs symmetric effective weights, a smooth stable branch and a small nudge ([certificate](settling.md#equilibrium-propagation-scope)). |
| `Brain` (composed) | One recurrent graph, with System 2 observers in the same equations and held trace, record and recall input. | `act`, `predict`, `accuracy` and `wait` need the full potential/adaptation residual below `learning.tolerance` within `learning.free_steps`, with one rescue on the damping ladder before the answer is refused ([qualification and refusal](start.md#qualification-and-refusal)). | The learner's contrast and the actor-critic's reward eligibility (always finite phases, `reward.eligibility_steps`), plus explicit trace and synaptic-memory writes. |

### Component limits

| Component | Implemented operation | Limit |
| --- | --- | --- |
| `Brain` (composed) | Qualified settles for every answer, local contrast and reward learning, working trace, synaptic and opt-in episodic and recall memory | A qualified answer covers the present settle. Memory enters as held input and is written after the settle, outside the equations; behaviour across ticks needs [persistence tests](evaluate.md#test-coherent-behavior-over-time). |

A private `imagine` call is a component operation. The application defines what its inputs
and outputs mean and which future variables it generates or supplies; a learned simulator
needs its own validation, and a value evaluator inside an exact rules-based search is a
value evaluator.

To check a fixed-point claim, evaluate the declared equations at the returned state: finite
values and the largest relevant defect, including slow states when the model has them. A
small integration step can make activity barely move while the equation defect stays large.
A small residual shows the state satisfies the equations within tolerance; uniqueness,
stability and useful behaviour need their own tests.

Numerical qualification, parameter commitment and measured learning improvement are three
separate facts. A qualified update establishes neither retention on earlier tasks nor
generalization; [evaluation](evaluate.md) covers those measurements. Eligibility, reward
and source identity stay attached to the action that was actually executed.

## Brain and Learner

- A composed `Brain` answers only from a qualified settle (1024 sweeps, residual `3e-3`, with
  a half-step fallback, or `damping` halvings under `qualified=True`); a refused `act` issues no action and changes no state ([qualification and refusal](start.md#qualification-and-refusal)).
- `LearnerConfig` teaches with finite phases by default. `qualified=True` is the opt-in that
  gates each phase on the full equations, with at most `damping=3` dt halvings by default. Give it a
  `nudged_steps` budget comparable to `free_steps` ([qualified teaching budgets](settling.md#qualified-teaching-budgets)).
- RMS-normalized rates are covered in [rates under normalization](teach.md#rates-under-normalization).
- Apical normalization has its own held-gain learning law. The ordinary symmetric
  plain-sum EP gradient argument does not cover it, and `ep_structure` refuses apical
  graphs. Residual qualification still checks the actual declared equations.

## State and production integration

Keep live state, learned parameters, memory stores, optimizer and step-size state, random
state and accounting separate, and save every one needed for an exact continuation,
including the boundary of a replay window. A custom callable policy needs its own
reproducible configuration; an array snapshot cannot serialize application code.

Calls validate shapes and finite values before mutating anything. An invalid call or a
failed transaction installs no parameter and no record. Counters of attempted work may
advance on a refused update; report them as attempted work, never as accepted learning. An
`imagine` call does not teach or alter live activity, though it can count its work.
With a learned code, a refused action, wait, lesson or outcome also restores the code
region's numerical step and previous phase. Invalid feedback is checked before that
region settles. An outcome already accepted before a later action refuses stays accepted.

`Connectome` arrays and its population mapping are read-only. `NeuralGraph.efficacy`,
`bias`, `log_gain` and the effective `weights` return read-only views. Replace parameters
in one of two ways:

- assign a complete array to `efficacy`, `bias` or `log_gain`; the setter validates it and
  refreshes the running backend;
- call `graph.with_parameters(efficacy=..., bias=..., log_gain=...)`, which returns a new
  graph on the same connectome; an omitted argument keeps the current array, and a new
  `log_gain` rebuilds every derived weight.

Do not mutate parameter or connectome arrays in place: derived transport caches assume
their values stay fixed. For topology changes, build a new `Connectome`; for population
changes, call `connectome.with_populations(name=indices)`, which returns a new connectome
with the named populations merged in. After a device update, always read `learner.brain`:
an older `NeuralGraph` object can share the updated device kernel while holding stale host
copies. Other model families change parameters through their own `set_parameters` or update
interface. Pin the library revision and backend for reproducible results and run a parity
check when changing device or precision ([backends](backends.md), [API](api.md),
[receipts](evaluate.md#receipts)).

## Recursive observation

An observer belongs to the same equilibrium as the system it observes only when its state
and feedback enter the same coupled equations and stopping check;
`Brain.compose(observers=...)` builds one ([compose a brain](tune.md#brain)).
