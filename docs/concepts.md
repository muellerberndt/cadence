# Concepts of the settling brain

This guide describes the neuron model, settling and the graph compositions of
the settling brain: `NeuralGraph`, `Learner` and `Records`. [Compose a brain](brain.md)
and [write a cortex](cortex.md) build one; [temporal learning](temporal.md) and
[the record patch](record-patch.md) are the path-based patches.

A Cadence brain is software state organised into neurons, synapses and regions,
with readouts, records and a learning update. It is a design for computation. It
does not ascribe experience to the software or establish a biological brain model.

For the application frame, start with [one continuing equilibrium brain](world-model.md):
bootstrap useful relationships and memory, use them, repair witnessed failures,
then continue the same life. The equations and isolated controls below describe
mechanisms within that design.

This is a self-reading computational design: local state, declared
synaptic boundaries, readback, records and feedback, with protocols and source-bound
receipts for evidence. The biological names describe computational roles.

The [experience guide](experience.md) starts from an ongoing learning life, connecting
world prediction, episodes, goals, action and communication. The
[composition guide](experience.md#connect-functions-through-actual-ports) connects those functions through actual ports.
The architecture is a hypothesis to test.

## Four principles

In this composition, Records learn what follows a reading, while local repair
and checked convergence describe the settled graph regions. Random drive
perturbations can propose alternative states. These are distinct from the
centered free/nudged contrast used for learning and
[temporal input planning](planning.md).

**Records.** A records cortex (`cd.Records`) subtracts each input unit's running mean from
a reading, maps it through a fixed random expansion onto many cells, keeps the most active
few and inhibits the rest. Each active cell holds one record per predicted field. The
prediction is the sum of the records the reading touches, weighted by activity, and
learning writes the witnessed outcome into exactly those records by the delta rule, at a
slow rate for consequences and a fast rate for valued fields such as reward. The number of
records a reading touches sets the learning speed: a sparse code confines each write to few
records, readings on other cells keep their reads, and records learn from one stream
without replay. Reading and writing need no settling. See [records](memory.md#records).

**Local repair.** Each neuron reads its own potential, its synaptic input and its drive,
and moves toward the input plus drive and bias, with adaptation when enabled.
`brain.settle_batch(drive,
steps=..., tolerance=...)` runs that update for every neuron of a batch; `brain.settle(stimulus)`
does the same for one stimulus. `Learner.step(drive, labels)`
runs a free phase and nudged phases on the same brain, then changes each synapse from the
activities of its own two neurons in those phases. Output nudges can read a target
and an output group; reward learning broadcasts a prediction-error signal. These
teaching signals are explicit. No backward computation graph is stored through the brain.

**Checked convergence.** A step cap is a work limit; the fixed-point
equations decide. `brain.residual(drive, state)` returns the largest equation error per row.
`brain.equilibrate(drive, budget=..., chunk=..., tolerance=...)` settles until every row's
residual is below the tolerance or the budget is spent, and returns the state with `steps`,
the per-row `residual` and the per-row `converged` flags. Pass the same `mask` and `nudge`
to a settle and to its residual check. A spent budget is a result to report.

**Random drive perturbations.** A settled brain gives one answer under one drive. To sample
alternatives, add bounded random drive to the latent neurons and settle again. Each batch
row supplies a separate candidate; check whether it reaches an equilibrium:

```python
import numpy as np
import cadence as cd

connectome = cd.layered(4, 16, 4, density=1.0, seed=1)
brain = cd.NeuralGraph(connectome, cd.learning_neuron_model())
latent = list(connectome.populations["hidden"])
rng = np.random.default_rng(0)

drive = np.zeros((8, connectome.n))          # eight rows, the same input in each
drive[:, list(connectome.populations["input"])] = np.eye(4)[0]
drive[:, latent] += rng.uniform(-0.1, 0.1, (8, len(latent)))  # bounded detuning
result = brain.equilibrate(drive, budget=512, chunk=32, tolerance=1e-6)
print(result.converged.all(), result.state.activation[:, list(connectome.populations["output"])].round(3))
```

Each added drive lies in `[-0.1, 0.1]`; a Gaussian draw has no such bound.
Check residuals, then rank candidates with a validated
critic or readout. Detuning does not teach a model of consequences; only a model
with useful learned or supplied dynamics can make these predictions useful for action.
[Evolve a brain](evolution.md#detuning-inside-a-life) ranks detuned candidates by a read of
the records or by a critic.

## Neurons and synapses

A *neuron* holds a potential `v` and publishes an activation `s`. It is a graded
(rate) unit and can stand for one biological neuron or, at coarse resolution, a
population. A directed *synapse* carries one neuron's activation to another. A
*reciprocal synapse pair* is a synapse and its reverse sharing one learned weight
(`Learner(reciprocal=True)`, the default). `Connectome` declares the synapses and
named populations of neurons, such as `input` and `output`.

`NeuralGraph` holds the connectome, the neuron model and the parameter arrays. On each
step it collects every neuron's synaptic input, then each neuron updates its own
state:

```text
synaptic_input[i] = sum(weights[e] * s[pre[e]] for synapses e ending at i)
v[i] += dt * (synaptic_input[i] + drive[i] + bias[i] - strength * a[i] - v[i])
s[i]  = activation(v[i])
```

Here `dt` is the numerical relaxation fraction, and `steps` counts updates of
this map. Neither quantity supplies elapsed environment seconds or biological
neural time without an explicit model and calibration. Environment events,
settling work and learning updates have [separate clocks](equilibrium-world-models.md#three-clocks).

The adaptation term is absent unless enabled. The effective weight of synapse `e`
is `gain * count[e] * efficacy[e] * exp(log_gain[pre[e]])`, where `count` is the
number of synaptic contacts. The synaptic `efficacy` starts from the connectome's
signs and can be learned. Transport follows the declared synapses; the neuron update
reads its own state and synaptic input. Softmax and key normalization additionally
read their declared groups.

A *stimulus* supplies the external drive, which is added to the neuron's input on
every step. The stimulus does not clamp the membrane potential, so synaptic input,
bias and adaptation still move a stimulated neuron.

## Settling and equilibrium

Named functional regions can share one connectome. Neuron-local updates cross the
synapses between regions and seek a common fixed point of the whole brain under the
current drive. Agreement means satisfying these coupled equations, not making
all neurons equal. The drive includes the memory readback held for this solve;
the full life can change that context and its parameters between solves.
Settle the brain once for that joint state; independent solves
followed by a merged visualization do not couple the regions. See [composition](experience.md#connect-functions-through-actual-ports).

Settling runs the update from rest or from a supplied state. A returned
`BrainState` can be a transient, a fixed point, or part of an oscillation.
`steps` limits work and `tolerance` stops on small activation movement;
`BrainState.activity_change` reports the total movement per row.
`NeuralGraph.residual` separately measures the remaining fixed-point equation
error. Saturation can produce small movement with a large residual.

An equilibrium need not be unique or stable. Carrying a state between inputs can
save work or select a different attractor. Test both cold and warm starts as
declared controls. Preserve context through a continuing life; reset transient
state only under the application's explicit terminal or independent-control
contract. A unique attracting equilibrium
under a fixed drive erases its initial condition; keeping history then requires
an explicit record, a trace, or a drive that carries history.


When every neuron's absolute incoming effective weight sum is below `1 / L`, with `L` the largest slope of the activation, one settling step is a contraction and the [certificate](certificate.md) bounds the remaining distance to the unique equilibrium from the last step's movement. `cd.certificate(brain)` reports it.

## Activation and adaptation

`NeuronModel` rebases a sigmoid so that zero potential emits zero. With zero
drive, zero bias, and zero adaptation, the all-zero state is an exact fixed
point. `learning_neuron_model()` uses a gentler slope and a small negative leak to
give the learner a responsive starting point. Neither setting guarantees convergence.

Builders start with zero bias. With signed random inputs, many hidden neurons can
remain near rest and carry weak learning contrasts. Measure activation and contrast
distributions for the intended experience stream. An explicit `NeuralGraph(..., bias=...)`
can shift the operating point; validate responsiveness and saturation together.

`Adaptation` adds one variable per neuron that follows its activation and
subtracts from its drive. In suitable mutually inhibitory circuits this can
produce an oscillation; [the dynamics tests](../tests/test_cadence.py)
check that behavior. Leave adaptation off when beginning with
equilibrium learning; its gradient interpretation needs additional assumptions.

## Three state lifetimes

| State | What changes it | How to manage it |
|---|---|---|
| Potential, activation, adaptation | Settling steps | Pass `state=` to continue; omit it to start from rest |
| A trace or fast-memory matrix | Explicit activity or observation updates | Preserve within a continuing stream; declare any terminal/control resets and preserve batch row identities |
| Learned weights, biases, records and consolidated associations | Local learner updates, reward/eligibility, `Records.write` or `SynapticMemory.observe` | Save each owning component; evaluate on a separate snapshot |

A [trace](api.md#streams-cadencestream) retains fading activity.
[Fast memory](memory.md) retains associations between supplied keys and values.
[Learning](learning.md) changes a reusable response through free/nudged endpoint
contrasts. Records change by one delta-rule write per witnessed outcome, without
settling. The centered learner uses three phases: free, positive nudge, and
negative nudge. Under its equilibrium assumptions, the small-nudge contrast
corresponds to a loss gradient with the stated parameter scaling. A raw `NeuralGraph`
does not update its weights because time passes or because it is settled again.
`Brain.step` schedules real-action learning; custom compositions own their
update clocks. A unique fixed point alone cannot preserve all past observations.

## Evidence and controls

A [protocol](protocols.md) declares stimuli, readouts, interventions, and predicates.
A shuffled connectome tests whether a response depends on the particular connections
under the same neuron model. It is one control.
High gains can saturate an excitatory circuit, so protocols can limit the active
fraction during gain selection.

`conformance` compares a trajectory with the neuron-by-neuron reference.
It checks the tested transport and update. A [receipt](receipts.md) binds stored
results to sources when those files are included and checked; its caller supplies
the arithmetic verifier. Neither check establishes benchmark fairness.

## Compared with backprop networks

| | Feed-forward model trained by backprop | Cadence brain |
|---|---|---|
| Reading an answer | Evaluate layers | Repeated neuron updates with a bounded solve and an optional residual check |
| State between inputs | A cache or a separate memory | A settled state, trace or record, each explicit |
| Credit | Reverse-mode differentiation | Free and nudged endpoint contrasts; delta-rule writes for records |
| Exact gradient conditions | Differentiable computation | Stable smooth equilibrium, symmetric effective weights, converged phases, vanishing nudge |
| Work | Forward and backward passes | Every step of the free and nudged phases plus the update |

Measure settling, learning and record maintenance separately on the same held-out
data, and include the simple algorithmic solver when a task has one.

This table compares update mechanisms.
Networks trained by backpropagation can also be recurrent, online, model-based and
memory-using. Matrix multiplication is an implementation operation. The experience
tests ask what the system acquires and transfers, how it uses goals and memories,
and what changes when its environment changes.

## When to add machinery

Settling, memory and learning are distinct operations of one ongoing step because their
state lifetimes differ. A new mechanism needs a failure it fixes and an ablation on the
same inputs, budget and seeds, with its state and cost counted. Brain functions
are compositions of the existing operations; [experience](experience.md) connects them.
