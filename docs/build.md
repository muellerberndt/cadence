# Build your own brain

This page trains, evaluates and saves three supported mechanisms: a record patch,
a supervised neural graph and a temporal model. These component recipes test
their own learning and memory contracts. For a continuing brain, start with the
[world-model lifecycle](world-model.md) and the [Brain quickstart](quickstart.md),
then retain that same brain through use, disturbance and correction. The recipes
below do not measure that complete lifecycle.

Every block runs on NumPy; the blocks of one section run together, in order.
Read [Cadence for machine-learning people](orientation.md) if the terms are new.

These guides use Cadence 0.81.0. Install the published package:

```bash
python -m pip install cadence-net==0.81.0
```

## A record patch on a stream

The record patch learns from a stream of moments: at each moment it hears
something and predicts an outcome. Use it when single facts must be kept after one
exposure, when a regularity should move into the weights, and when the data will be
gone before the brain has learned them.

### 1. Encode

Paths have shape `(batch, time, ports)`. Batch rows are independent streams sharing
parameters. Encode a categorical input as one-hot ports and give a continuous input
unit variance. A categorical outcome is one-hot and declared in `groups`, so the slow
readout ends in a softmax per group and learns by cross-entropy; a continuous outcome
is a plain port.

The data here are streams of tokens from a vocabulary of eight, and the outcome at
each moment is a rule of the token heard. Replace `make_streams` with your own loader
that returns the same shapes.

```python
import numpy as np
from cadence import RecordPatchNet

VOCAB, OUTCOMES, HIDDEN = 8, 4, 16
rng = np.random.default_rng(3)


def make_streams(count, length):
    tokens = rng.integers(VOCAB, size=(count, length))
    return np.eye(VOCAB)[tokens], np.eye(OUTCOMES)[tokens % OUTCOMES]


heard, outcome = make_streams(32, 16)             # 32 training streams of 16 moments
heard_test, outcome_test = make_streams(8, 16)    # 8 held-out streams
assert heard.shape == (32, 16, VOCAB) and outcome.shape == (32, 16, OUTCOMES)
```

### 2. Size it

```python
net = RecordPatchNet(
    VOCAB, HIDDEN, OUTCOMES, seed=1,
    cells=4096, active=16,        # the store: 4,096 cells, 16 light per reading
    record_rate=1.0,              # a reading written once takes its outcome whole
    groups=(OUTCOMES,),           # one categorical port of four outcomes
    slowest=8.0,                  # context channels from two to eight moments at birth
)
```

| Knob | What it sets | Where to start |
| --- | --- | --- |
| `hidden` | context channels; the width of the slow patch | 16 to 128 |
| `cells`, `active` | the store's size and how many cells a reading touches | 4,096 and 16 for a toy; 65,536 and 32 to 64 for a corpus; enough active cells that neighbouring readings overlap |
| `record_rate` | how much of the residual a write stores | 1.0 when each reading is written once; `record_averaging=True` for a store written hundreds of times per cell |
| `slowest` | the longest initial retention timescale, in moments; the gates learn from there | a few times the longest dependency you expect |
| `groups` | categorical ports, one softmax per group | every symbol output |
| `record_width` | a sign code narrower than a wide categorical port | hundreds of columns for thousands of categories |
| `record_writes` | `"sequential"` or `"batch"` writes within one call | `"batch"` when a call holds many moments that share cells |
| `rate` (in `observe`) | the slow step; the loss is a mean over time and outputs, so the useful rate scales with their product | 1 to 10 here; thousands for a wide picture port |

### 3. Evaluate before, during and after

Score the brain two ways: with its records, and with the slow weights alone. Score on
held-out streams. Compare with the simplest control the task admits; here it is the most
frequent outcome of the held-out streams.

```python
def accuracy(net, heard, outcome, records=True):
    probe = net if records else RecordPatchNet.restore(net.snapshot())
    if not records:
        probe.records.tables["y"][:] = 0.0                 # the same weights, an empty store
    said = probe.imagine(heard, state=np.zeros((len(heard), HIDDEN))).output.argmax(-1)
    return float((said == outcome.argmax(-1)).mean())


chance = np.bincount(outcome_test.argmax(-1).ravel()).max() / outcome_test[..., 0].size
before = accuracy(net, heard_test, outcome_test, records=False)
```

### 4. Learn by day

With the world present, learn at a slow rate above zero and let every observed moment
write its residual into the store. `backtrack=True` admits a step only when a
target-free replay lowers the slow loss, halving the rate until it does. Reset the
context at the start of every independent pass.

```python
for day in range(20):
    net.reset()
    seen = net.observe(heard, outcome, rate=8.0, backtrack=True)
    assert seen.updated or seen.reason == "no_decreasing_parameter_step"

with_records = accuracy(net, heard_test, outcome_test)
slow_only = accuracy(net, heard_test, outcome_test, records=False)
print(f"held-out: chance {chance:.2f}, before {before:.2f}, records {with_records:.2f}, slow weights alone {slow_only:.2f}")
assert slow_only >= 0.95 and with_records >= 0.95
```

The records score the held-out streams from the first passes, by overlap: a held-out
reading touches the cells of the training readings that heard the same token, and their
records agree. The slow weights take longer and then know the rule without the store.
A rule about what came before, a conjunction of this token and the last, is something
the gate has to learn and takes many more updates; a second patch that reads the first
patch's context learns it ([two patches in depth](record-patch.md#two-patches-in-depth)).

### 5. Keep a fact after one exposure

Hear the held-out streams once at slow rate zero. Nothing but the store changes, and the
store recalls every moment:

```python
net.reset()
first = net.observe(heard_test, outcome_test, rate=0.0)
assert first.writes == heard_test.shape[0] * heard_test.shape[1]
net.reset()
assert accuracy(net, heard_test, outcome_test) == 1.0
```

### 6. Sleep when the data are gone

When the streams are gone and the store is the only copy, a night moves
what the store holds into the slow weights: `sleep` dreams every cue once (the store's
own completion of it), teaches the slow weights those fixed dreams with `observe(write=False)`,
and at dawn writes the dreams back so the store holds only what the weights did not take.

```python
sleeper = RecordPatchNet(VOCAB, HIDDEN, OUTCOMES, seed=1, cells=4096, active=16,
                         record_rate=1.0, groups=(OUTCOMES,), slowest=8.0)
for _ in range(2):                                     # the day: records only
    sleeper.reset()
    sleeper.observe(heard, outcome, rate=0.0)
bedtime = accuracy(sleeper, heard_test, outcome_test, records=False)
night = sleeper.sleep([heard], passes=40, rate=8.0, backtrack=True)
dawn = accuracy(sleeper, heard_test, outcome_test, records=False)
print(f"slow weights alone, held-out: bedtime {bedtime:.2f}, dawn {dawn:.2f}; {night}")
assert dawn >= 0.95
```

A learner with the world present learns by day; a learner whose day is over sleeps.
What the store reads at bedtime bounds the night: dreams carry the store's errors as
faithfully as its regularities ([acquisition in two phases](record-patch.md#acquisition-in-two-phases-records-by-day-weights-by-night)).

### 7. Save, load, read back

```python
path = net.save("stream_brain.npz")
restored = RecordPatchNet.load(path)
np.testing.assert_array_equal(
    restored.imagine(heard_test, state=np.zeros((8, HIDDEN))).output,
    net.imagine(heard_test, state=np.zeros((8, HIDDEN))).output,
)
readback = net.readback()
print(readback.updates, readback.writes, readback.record_entries, readback.parameter_revision)
```

The checkpoint holds the parameters, the record tables, the running statistics of the
reading and the live context. `readback` is the detached state with its counters; an
application may feed it to another patch through a port, and it is never a teaching signal
by itself.

### 8. Diagnose the store

A store can fail silently, because it always holds something. Replay the training
readings through `net.records.code(reading, valued=False)` and count, per cell, how often it
is active: the cells in use, the share of activations the most active cells take, and the
code overlap when only one block of the reading changes are the three numbers to watch
([diagnosing a record store](record-patch.md#diagnosing-a-record-store)). If the reading
concatenates blocks of different scale, the larger block owns the address; give every unit
unit variance.

### 9. Grow it

- Pictures: read the grid through a tied local kernel at the port,
  `RecordPatchNet(port=StructuredPort(...))` ([maps](record-patch.md#maps-a-structured-input-port)).
- Depth: `RecordPatchStack` for a conjunction of what came before.
- Several senses: `JointRecordPatches` settles several patches as one equilibrium through
  declared ports ([several patches joined by ports](record-patch.md#several-patches-joined-by-ports)).
- A belief carried under action, with imagination: [the belief patch](belief.md).
- A brain in a page: [record the settlings](pages.md) for an application to draw.

## A supervised graph-learning control

This independent-row classifier tests custom wiring and graph plasticity. A
genome names regions and projections, `develop` creates one connectome,
`NeuralGraph` settles it, and `Learner` updates synapses using free and nudged
phases. The graph has reciprocal feedback; arranging regions in a chain does
not make its settlement a feed-forward pass.

The example measures held-out classification and checks saved predictions.
It does not exercise continuing context, memory recall, interference or recovery.
Use [Brain.compose](brain.md) for the continuing interaction route.

### 1. Encode

A drive has shape `(batch, n)`, one column per neuron of the connectome; the features go
on the input region's neurons. Scale features to roughly unit range. Labels for
`Learner.step` are integer indices into the output group.

```python
import numpy as np
import cadence as cd
from cadence.regions import cortex, motor_cortex

FEATURES, CLASSES = 8, 3
rng = np.random.default_rng(0)
centres = rng.standard_normal((CLASSES, FEATURES)) * 1.5


def make(count):
    labels = rng.integers(CLASSES, size=count)
    return centres[labels] + 0.6 * rng.standard_normal((count, FEATURES)), labels


x_train, y_train = make(240)
x_test, y_test = make(120)
```

### 2. Wire it

```python
genome = cd.Genome(
    regions=(cd.Region("features", FEATURES), cortex(24), motor_cortex(CLASSES, lateral=-0.5)),
    projections=(
        cd.Projection("features", "association", reciprocal=False),   # senses drive the cortex one way
        cd.Projection("association", "motor"),                          # reciprocal: the nudge travels back
    ),
)
connectome = cd.develop(genome, seed=0)
brain = cd.NeuralGraph(connectome, cd.learning_neuron_model())
features = list(connectome.populations["features"])
learner = cd.Learner(brain, connectome.populations["motor/actions"], cd.LearnerConfig(eta=1.0, beta=0.1))


def drive(x):
    d = np.zeros((len(x), connectome.n))
    d[:, features] = x
    return d
```

Hidden neurons learn only where a nudge on the outputs reaches them, so the projection
from the cortex to the motor region is reciprocal. `lateral=-0.5` makes the actions
compete. [Write a cortex](cortex.md) has the catalogue of regions, projections and heads;
[compose a brain](brain.md) adds working trace, associative memory and actual-outcome
reward feedback around the neural graph.

### 3. Calibrate, learn, evaluate

`calibrate` picks the sampled gain whose mean free motor activity is closest to
0.5; it may still leave the readout unresponsive. Its default search is centered
on the current gain, and an explicit `grid` fixes the candidates and their order.
Inspect `learner.last_calibration` for the residuals and all attempted work.
Qualified learning also requires each calibration candidate to meet the full
equations before it can be selected. Choose the gain on training inputs before
scoring held-out data ([calibration contract](learning.md#calibrating-the-operating-point)).
This example declares a small-gain grid for its densely connected motor region;
the widest operating-point search does not necessarily select the best learner.
Each `step` runs a free phase and two nudged phases and moves the synapses once;
the configuration in this example uses finite teaching.

```python
learner.calibrate(drive(x_train[:32]), grid=[0.02 * 1.3**k for k in range(16)])
before = learner.accuracy(drive(x_test), y_test)
for epoch in range(3):
    order = rng.permutation(len(x_train))
    for start in range(0, len(x_train), 24):
        rows = order[start:start + 24]
        learner.step(drive(x_train[rows]), y_train[rows])
    print(epoch, "held-out accuracy", round(learner.accuracy(drive(x_test), y_test), 3))
assert learner.accuracy(drive(x_test), y_test) >= 0.95
```

### 4. Check the settling

A converged phase satisfies the neuron equations at its state; the residual is the check.
The certificate bounds the distance to the equilibrium when the incoming weight mass of
every neuron is below its limit; a learned brain often crosses it, and then the residual
is the check.

```python
free = learner.free(drive(x_test[:8]))
print("largest equation residual per row", learner.brain.residual(drive(x_test[:8]), free).round(4))
cert = cd.certificate(learner.brain)
print("certified", cert.certified, "row mass", round(cert.row_mass, 2), "limit", cert.mass_limit)
```

### 5. Save, load, draw

```python
from pathlib import Path

learner.save("decision_brain.npz")
again = cd.Learner.load("decision_brain.npz")
assert again.accuracy(drive(x_test), y_test) == learner.accuracy(drive(x_test), y_test)

```

To draw it, record the settlings of a solve and hand them to a page of your own
([the brain in a page](pages.md)).

### 6. Grow it

- Reward instead of labels: `ActorCritic` over the same learner, eligibility traces and a
  broadcast prediction error ([reward](reward.md)); `Brain` is the ready composition
  with a critic, a working trace and associative memory ([compose a brain](brain.md#brain)).
- Learned consequences and planning: a separate temporal model
  ([learn consequences, then act](interaction.md)); this is not automatically
  integrated into `Brain.compose`.
- Let selection size the regions: `evolve` over the genome across lives
  ([evolve a brain](evolution.md)); any hand-set constant can be a gene.
- A measured connectome: `Connectome.from_synapses` from a synapse list, then a
  protocol with a shuffled control ([protocols](protocols.md)).
- Speed: `NeuralGraph(..., backend="torch")` for a large blocked connectome
  ([backends](backends.md)).

## A temporal patch on continuous observations and actions

The temporal patch models a body: continuous observations and actions on
paths of shape `(batch, time, ports)`, a learned dynamics model, and planning that
repairs the action ports under that model.

[Learn consequences, then act](interaction.md) is the complete walkthrough: collect
executed actions and measured positions, learn the transition with `observe` on
independent episodes, check the forecast error on held-out paths against a persistence
baseline, then `plan` under a goal, execute the first action, measure, replan, through an
unannounced disturbance. It runs in a few seconds and asserts its own numbers. The order
of the calls is the recipe for any body:

1. Declare the ports: which are observations, which are actions, and a presence flag
   where an observed zero must differ from no observation.
2. `net.reset()` between independent episodes; `observe(inputs, measured, beta=0.01, rate=...)`
   on executed inputs and measured outcomes only.
3. `imagine(inputs, state=...)` on held-out paths; report the forecast error and the
   persistence baseline.
4. `plan(inputs, goal=..., controls=..., bounds=..., state=...)`; execute the first
   action through the real body; admit what was measured; plan again.
5. Protect a response before further learning with `TemporalMemory.protect` and learn
   through `memory.observe` ([response protection](temporal-memory.md)).

The solver is dense in the hidden width (`N * T * H^3` per detuned chain), so keep the
hidden width and the path length modest and read [scaling](scaling.md) before a long
run. For a belief that a transition carries under action from pictures, with
imagination that consumes no observation, use [the belief patch](belief.md).

## Before you trust a number

- The score is on streams, episodes or rows the brain never saw, and the split was made
  before any tuning.
- The control is stated: the majority outcome, persistence, a linear predictor, the
  search alone, a shuffled connectome.
- The slow weights and the records are scored separately.
- The full equation residual meets the declared tolerance for a qualified settle;
  movement alone is not that check. Certificates retain their own hypotheses, and
  a reported `reason` other than an update is read, never ignored.
- The seed, the library version and the data revision are recorded; a
  [receipt](receipts.md) binds the numbers to the sources.
- [Common missteps](missteps.md) lists the ways a good-looking number is wrong.
