# Records: read before writing

A brain with bounded state needs records it can read back and correct.
[`cd.Records`](#records) is the records cortex: a fixed sparse expansion of a reading and one
delta-rule table per predicted field, at two rates. `FastSynapses` stores a key-to-value
matrix of fast synaptic weights per stream. Its storage is fixed by the key and value
widths; it does not grow with the number of observations. Slow weights can learn
representations around that store, but the store does not learn its own keys or decide
when an observation is trustworthy.

For repeated and salient experiences, [`SynapticMemory`](continuous.md) adds persistent
shared synapses and fading per-stream residuals. `Brain.compose` includes this
consolidation rule; `episodic=False` omits that pathway. The `FastSynapses` API below
provides independent stream records and an immediate residual-write rule.

<a id="three-kinds-of-memory-in-brain"></a>

## Memory in the composed brain

| Mechanism | What it retains | What changes it | What survives `brain.reset()` |
| --- | --- | --- | --- |
| Working `Trace` | Recent association-region activity used as context | Each admitted live action | Nothing; the trace is cleared |
| `Efference` copy (opt-in gene) | The fading one-hot of the commands the stream issued | Each admitted live action | Nothing; the copy is cleared |
| Graph plasticity | Learned synaptic parameters and biases | Supervised contrasts and reward-modulated eligibility | Learned parameters |
| `SynapticMemory` | Cue-to-outcome associations, with fast per-stream `F` and shared persistent `C` | Observed chosen-action outcomes | Both `F` and `C` |

These are functional software mechanisms. Their capacities follow their state
dimensions, coding and update rules; a human working-memory estimate does not set
the number of durable associations they can retain. A lost response can come
from a missing trace, interfering parameter updates or interfering record writes.
Test the mechanism that was used to acquire the response.

`predict` and `accuracy` ignore the working trace, the efference copy and the
associative store.
They qualify the current learned graph's independent response. `act` reads the
available memory pathways before qualifying the full graph state. To test consolidated
recall, reset live state and clear only fast residuals before each free query:

```python
import numpy as np
from cadence import Brain

remembering = Brain.compose(inputs=4, actions=2, modules=(16, 8), seed=7)
remembering.reset()
assert remembering.hippocampus is not None
remembering.hippocampus.reset(1)  # Clear F; retain persistent C.
answer = remembering.act(np.array([[1.0, 0.0, 0.0, 0.0]]), greedy=True)
assert answer.shape == (1,)
```

`teacher=` teaches graph parameters without inserting its label into the store.
The normal reward loop stores the actual chosen action's observed reward, not an
invented reward for a teacher label. Direct `observe` calls are an explicit
key/value acquisition rule; they test that store rather than graph plasticity.
An associative read is a linear drive into the graph; the resulting action must
still qualify the complete neural equations.

Measure acquisition first, then expose the same brain to competing experience
and query old and new responses without teachers. Rehearsal can protect recall
and must be counted as additional teaching or record writes. Finite storage and
successful bounded recall do not establish lifelong retention.

Working-trace amplitude and decay are task-dependent settings. A stronger trace
retains earlier context and can also interfere with acquiring a new relation,
and neither a smaller amplitude nor a different decay is a general fix. Where
the previous moment carries nothing the decision needs, the trace is
interference the actor has to learn away: on a continuing contextual bandit the
composed default amplitude held the hit rate at chance over several seeds while
amplitude 0.0 reached 1.0, and a small amplitude with a smaller actor rate also
reached 1.0
([night-replay chamber](../benchmarks/replay/README.md#the-working-trace-on-a-continuing-bandit)).
The [vanished-cue instrument](../benchmarks/recall/README.md) measures a
continuing brain against checkpoint-matched trace interventions and an
explicitly trained history comparator; its software checks validate the
instrument, not a recall horizon. For demonstrations without a reward protocol,
use the [supervised-only stream](continuous.md#a-supervised-only-stream).

## One correction

For a unit key `k`, value `v`, and matrix `M`:

```text
prediction = k @ M
error      = v - prediction
M         += rate * outer(k, error)
```

With `rate=1`, `amplitude=1` and a nonzero key, reading that key immediately after the write returns
`v`, to numerical precision. Repeating the same correct observation has zero residual
and does not accumulate strength. For an earlier key `q`, the change in its read is
`rate * dot(q, k) * error`: orthogonal records are preserved, correlated ones interfere.
This identity explains both the useful behaviour and its limit.

Cadence normalizes delta keys and queries to unit length. Normalization reads the whole
key vector; each fast synapse then updates from its presynaptic key coordinate and its
postsynaptic neuron's error. This is normalized delta learning, closely related to
normalized LMS and [delta-rule fast weights](https://arxiv.org/abs/2406.06484).
A read is one linear transport from key to value; it involves no softmax attention and
no settling. `cd.Records` writes by the same rule, with a unit-length sparse code in the
place of the key.

## Records

A records cortex keeps records of what followed each reading it took part in. Its
prediction for a reading is the sum of the records that reading touches, each weighted by
the activity of its cell. Learning writes the witnessed outcome into exactly those records,
by the error at that reading, and touches nothing else. The code decides which readings
share records: two readings share the records of the cells active in both.

`cd.Records` is that learner in four parts. A mean-free reading: each input unit's running
mean is subtracted, so the code follows what deviates from the usual reading and a small
cue can move it. A fixed sparse expansion: a random projection with fixed offsets maps the
reading onto many cells, the `active` cells with the strongest drive keep their rectified
drive, the rest are inhibited to zero, and the code is scaled to unit length; the
expansion is never learned. Delta-rule records: one table per predicted field, one record
per cell, read through the active cells and written by the error. Two rates: `rate` for
the consequence fields, so a regularity is averaged over many outcomes, and `valued_rate`
for the valued fields such as reward, so one exposure writes.

The number of records a reading touches sets the learning speed. A write at one reading
moves the read at another reading by the overlap of their codes times the correction. A
sparse code touches few records: at rate one, a write reproduces the outcome at its reading
exactly, and later writes change it only through shared cells. A dense code touches many:
each outcome is averaged into all of them, and later outcomes overwrite it. A learner whose
every parameter takes part in every prediction moves every prediction with each update,
which a rehearsal of decorrelated real observations counters. Records learn
from one stream in its own order, as the [records tests](../tests/test_records.py) do with
a correlated walk.

The settled regions keep the work that needs a fixed point: completion of a partial
reading into a consistent state, context carried across a delay, and the policy where
credit arrives after other decisions. Writing a record needs no settling; reading one is a
product of the code with each table. The free/nudged rule and the eligibility trace belong
to the settled regions; the [certificate](certificate.md) and the residual check bound
their settling and say nothing about the records.

### Codes, reads and writes

`code(readings, *, adapt=False)` takes `(batch, inputs)` readings, or one `(inputs,)`
reading, and returns `(2, batch, cells)`: the plain code, then the valued code.
`code(...)[:, 0]` is the `(2, cells)` code of one reading, the shape `write` takes.
`read(code)` returns each field's read, `(width,)` for one code and `(batch, width)` for a
batch; a consequence field reads the plain code and a valued field the valued code.
`write(code, targets, known=None)` moves each field named in `targets` toward its target
through the code of that field, returns the number of fields written and adds it to
`writes`. A field absent from `targets` keeps its records.

```python
import numpy as np
import cadence as cd

records = cd.Records(
    inputs=6, fields={"next": 3, "reward": 1}, valued=["reward"], cells=4000, active=40, seed=0
)
reading = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])    # one of three places, one of three actions
code = records.code(reading, adapt=True)[:, 0]        # a witnessed reading, shape (2, cells)
assert np.allclose(records.read(code)["next"], 0.0)   # read before writing
records.write(code, {"next": np.array([0.0, 1.0, 0.0]), "reward": np.array([1.0])})
after = records.read(code)
print(after["next"], after["reward"])                 # [0.  0.2 0. ] [1.]

other = records.code(np.array([0.0, 1.0, 0.0, 0.0, 1.0, 0.0]))[:, 0]   # another place
overlap = code[0] @ other[0]
assert np.allclose(records.read(other)["next"], 0.2 * overlap * np.array([0.0, 1.0, 0.0]))
```

The consequence field moved by `rate` (0.2) times the error and the valued field by
`valued_rate` (1.0). The read at `other` moved by the overlap of the two codes times the
correction; the records of every cell the first reading left inactive stay zero.

### The scale of a reading

The projection is scaled so that a reading with unit variance per unit gives
drives of unit scale, and the cells' fixed offsets (`bias`) are in those
drive units. A reading much smaller than that lets the offsets choose the
winners, and the same cells fire for every reading; a reading whose blocks
differ in scale lets the larger block own the address. When a reading
concatenates several sources, give each unit unit variance, then count the
cells in use and the share of the most active cells on real readings
([diagnosing a record store](record-patch.md#diagnosing-a-record-store)).

### Habituation

With `habituation` above zero (`1e-5` by default), `code(..., adapt=True)` first moves the
running mean `mean` toward each witnessed reading, and every code subtracts `mean` before
the expansion. `seen` counts the witnessed readings, and the mean moves at the rate
`max(habituation, 1 / seen)`: it is the plain average of the readings until one over their
count falls to `habituation`, and follows at that rate after. The mean settles, and a
settled mean keeps the code of a reading on the cells where its records were written. A
unit that carries the same value in every reading adds nothing to the code, and a small
deviation such as a cue moves it. `habituation=0` codes raw readings.

```python
def cue_overlap(habituation):
    cue = cd.Records(32, {"y": 1}, cells=2000, active=20, habituation=habituation, seed=5)
    first, second = np.full(32, 2.0), np.full(32, 2.0)    # the same background
    first[30], second[31] = 3.0, 3.0                       # a different cue unit
    for _ in range(500):
        cue.code(np.stack([first, second]), adapt=True)
    codes = cue.code(np.stack([first, second]))[0]
    return float(codes[0] @ codes[1])


print(round(cue_overlap(0.0), 2), round(cue_overlap(0.01), 2))   # 0.92 0.35
```

A reading component that varies between episodes without deciding the outcome, such as a
context trace, moves the code too and spreads the records of one situation over several
codes. Keep such components out of the reading; a store read that decides the outcome
belongs in it.

### Two codes and two rates

The plain code serves the consequence fields, written at `rate` (0.2 by default). The
fields named in `valued` read and write through the valued code at `valued_rate` (1.0 by
default). Without `pathways` the valued code is the plain code. `pathways` lists index
arrays into the reading, one per input pathway (the observation, the goal, the action, the
read of a store). The valued code divides each pathway of the mean-free reading by its
running norm, kept in `pathway_norm` and moved by witnessed readings at `pathway_rate`, so
a goal of one active unit has a say in the valued code comparable to a dense observation.
Consequence and value then read one reading through two codes.

```python
observation, goal = np.arange(12), np.arange(12, 14)
valued = cd.Records(
    14, {"next": 12, "reward": 1}, valued=["reward"], cells=4000, active=40,
    pathways=[observation, goal], seed=0,
)
rng = np.random.default_rng(0)
for _ in range(1000):                                  # witnessed readings
    x = np.zeros(14)
    x[observation] = rng.random(12) < 0.5
    x[12 + rng.integers(2)] = 1.0
    valued.code(x, adapt=True)
scene = (rng.random(12) < 0.5).astype(float)
codes = valued.code(np.stack([np.r_[scene, 1.0, 0.0], np.r_[scene, 0.0, 1.0]]))  # two goals
print(valued.pathway_norm.round(2))                    # the running norms of the two pathways
print(round(float(codes[0, 0] @ codes[0, 1]), 2))      # the plain codes of the two goals: 0.41
print(round(float(codes[1, 0] @ codes[1, 1]), 2))      # their valued codes: 0.13
```

A consequence field at a slow rate averages a regularity over many outcomes. A valued
field at rate 1.0 writes in one exposure: after one write, its read at that reading equals
the target.

### Task sets

`tasks` indexes the reading's task units, a goal port with one active unit. The cells are
then divided into one group per task unit, drawn from the seed, and a reading's valued
code takes its winners from the group of the unit with the largest value; a reading whose
task units are all zero draws from every cell. The plain code and the consequence records
stay shared by every task. A reward earned under one goal is written into cells no other
goal reads, so a value learned for one task survives the rewards of the tasks that follow.
`task_of_cell` holds each cell's group.

```python
tasked = cd.Records(
    12, {"y": 1, "value": 1}, valued=["value"], cells=2000, active=20, habituation=0.0,
    tasks=[10, 11], seed=8,
)
first, second = np.zeros(12), np.zeros(12)
first[:10] = second[:10] = np.random.default_rng(3).random(10)   # the same scene
first[10], second[11] = 1.0, 1.0                                # two tasks
codes = tasked.code(np.stack([first, second]))
print(np.array_equal(codes[0, 0], codes[0, 1]), round(float(codes[1, 0] @ codes[1, 1]), 2))   # True 0.0
tasked.write(tasked.code(first)[:, 0], {"value": np.array([1.0])})
print(tasked.read(tasked.code(second)[:, 0])["value"])                                          # [0.]
```

### Fan-in

`fan_in` gives each cell a random subset of that many pathways, drawn from the seed; the
cell reads those pathways and every input outside the pathways. A cell that reads the
action and the local scene without the place is active wherever that scene and action
recur, so the record it writes at one place is read at every other: the rule of a
consequence generalises across the inputs the cell does not read. Cells that happen to
read the place keep what is particular to it. The reads sum over both kinds of cell.

```python
scene, action, place = [0, 1, 2, 3], [4, 5], list(range(6, 22))
local = cd.Records(
    22, {"next": 4}, cells=4000, active=40, habituation=0.0,
    pathways=[scene, action, place], fan_in=2, seed=4,
)
reads = np.stack([(local.projection[p] != 0).any(axis=0) for p in (scene, action, place)])
print(reads.sum(axis=0).min(), reads.sum(axis=0).max())   # 2 2: every cell reads two pathways
print(int((~reads[2]).sum()))                            # the cells that never read the place
```

### Known entries

`known` maps a field to a boolean mask of the target entries that were observed. The error
of every other entry is zero, so an unobserved entry keeps its records.

```python
partial = cd.Records(5, {"y": 2}, cells=500, active=10, rate=1.0, seed=6)
code = partial.code(np.ones(5))[:, 0]
partial.write(code, {"y": np.array([1.0, 1.0])}, known={"y": np.array([True, False])})
print(partial.read(code)["y"])                         # [1. 0.]
```

### Imagined readings

A reading the learner imagines, a candidate action or a predicted next observation, is
coded with `adapt=False`: the running mean and the pathway norms follow witnessed readings
only. An imagined reading carries the same missing flags as the witnessed readings it
stands for. With habituation, a flag that witnessed readings always carry has a running
mean of one; an imagined reading that omits it deviates from the mean by a whole unit,
and its read goes through cells that witnessed readings never wrote. `code(readings,
valued=False)` computes the plain code alone and leaves the valued code unset (NaN), for
imagined readings whose consequence fields alone are read.

```python
flags = cd.Records(7, {"next": 4}, cells=4000, active=40, habituation=0.01, seed=2)
rng = np.random.default_rng(0)
for _ in range(2000):   # four places, the flag of a field the sensor never sees, two actions
    x = np.zeros(7)
    x[rng.integers(4)], x[4], x[5 + rng.integers(2)] = 1.0, 1.0, 1.0
    flags.code(x, adapt=True)
witnessed = np.array([0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0])
imagined = witnessed.copy()
imagined[4] = 0.0                                      # the missing flag left out
codes = flags.code(np.stack([witnessed, imagined]))[0]
print(round(float(codes[0] @ codes[1]), 2))            # 0.36: the imagined read uses other cells
```

### Cost and state

A read is one product of the code with each table and a write one outer-product step per
field, on NumPy in every backend. `parameters()` counts the record entries, `cells` times
the width of each field; `projection` and `offset` are fixed. `to_dict()` returns the
configuration that rebuilds the same cells from `seed`, and the learned state is `tables`,
`mean`, `seen`, `pathway_norm` and `writes` ([checkpoints](brain.md#checkpoints)).

## Reading and writing directly

```python
import numpy as np
import cadence as cd

memory = cd.FastSynapses(np.arange(4), np.arange(4, 6), rule="delta")
keys = np.eye(4)[:2]                   # two independent streams
values = np.array([[1., 0.], [0., 1.]])
before = memory.recall(keys)          # query before revealing the values
memory.observe(keys, values)          # one observation in each stream
assert np.allclose(memory.recall(keys), values)
memory.reset(2, rows=np.array([True, False]))  # only the first episode ended
```

`observe(key, value, write=None)` accepts two-dimensional arrays with matching batches;
a boolean `write` mask gates each stream. Every call decays all strengths by `decay`
before writing selected rows. Reads do not decay state. `rate` lies in `[0, 1]` in delta
mode. A zero key cannot write an association. `amplitude` scales reads as a drive,
without changing the target used by the write rule. Reading a different batch size returns
the empty-stream baseline without changing live records. Observing a different batch
size starts fresh streams; use `keep(rows)` when dropping streams while preserving
their identities.
`reset` clears records; `writes` counts lifetime write events.
An overflowing write raises `ValueError` and preserves the previous records and
separator state. Scale keys and values to the range your application needs.

## With an existing brain

The same memory can read key neurons and drive value neurons:

```python
connectome = cd.Connectome.from_synapses(
    6, pre=[0, 1, 2, 3], post=[4, 4, 5, 5], count=[1] * 4,
    populations={"key": range(4), "value": (4, 5)},
)
brain = cd.NeuralGraph(connectome, cd.learning_neuron_model())
drive = np.zeros((2, connectome.n))
drive[:, :4] = np.eye(4)[:2]

# pre/post are declared neuron indices in this connectome.
fast = cd.FastSynapses(
    np.array(connectome.populations["key"]), np.array(connectome.populations["value"]), rule="delta"
)
drive_with_record = fast.stimulate(drive)
state = brain.settle_batch(drive_with_record)
# Later, after observing what actually followed:
observed_rows = np.array([True, False])
observed_values = np.array([[1.0, 0.0], [0.0, 0.0]])
fast.update(state, write=observed_rows, post=observed_values)
```

The key used by `update` is the key neurons' *activation*, while `read`/`stimulate` uses the
key range of the supplied *drive*. Supply compatible representations; nonlinear
activation can change a dense key's direction. Use direct `observe`/`recall` when the
same external key should be used for both operations. Avoid writing the brain's
own prediction as if it were new evidence.

`update(state, write=None)` only decays the stored strengths, while
`observe(..., write=None)` writes every row. The caller owns episodic fast state;
`Learner.save` does not save a separate `FastSynapses` or `Trace` object.

## Additive memory

The default `rule="hebb"` adds `rate * k vᵀ` on each write.
`normalize=True` unit-normalizes keys and divides a read by decayed write mass.
`replace=True` clears matrix rows whose key coordinates are positive before adding the
association. That is useful for one-hot slots, but erases unrelated records with
dense positive keys. Both flags are rejected in delta mode to avoid mixing incompatible
read and write semantics.

Measure retention and correction on the same observed streams, including supplied
key/value parsing. Count mutable matrix storage as well as learned parameters. Fixed storage
is a capacity tradeoff: contradictory associations, nearly parallel keys, and more
independent values than the key rank cannot all be represented exactly.

## Pattern separation

A write at key `k` moves the read at key `q` by the dot product `q · k` times the correction,
so records interfere exactly as much as their keys overlap, and keys with disjoint supports
do not interfere at all. `PatternSeparator` turns correlated keys into sparse codes before
a `FastSynapses` or `SynapticMemory` store sees them: a fixed random projection onto a
wider range, then the strongest positive `winners` entries kept and the rest set to zero.
This can reduce overlap but does not guarantee different codes. With `center` above zero
the separator subtracts a running mean of observed keys first; it does not learn which
differences matter for a task. `cd.Records` builds its own expansion, with habituation and
two codes, for the records cortex.

```python
import numpy as np
import cadence as cd

key_neurons, value_neurons = np.arange(32), np.arange(32, 40)
sep = cd.PatternSeparator(inputs=32, expansion=1024, winners=8, seed=0, center=0.99)
sep.habituate(np.random.default_rng(0).standard_normal((256, 32)))  # the environment's keys
memory = cd.FastSynapses(pre=key_neurons, post=value_neurons, rule="delta", separator=sep)
```

`sep.code(key, learn=False)` returns the `(batch, expansion)` sparse code of `(batch, inputs)`
keys. `FastSynapses(separator=sep)` codes keys and queries with it and moves the running
mean on writes (`learn=True`). `habituate` sets the running mean from a sample of the
environment's keys before anything is stored; the slow `center` then tracks it. A fast
running mean can move codes between a write and its read; code overlap, finite capacity
and contradictory values can also impair recall.

The record then has `expansion` rows per stream instead of `inputs`, which is the price:
storage grows with the code, capacity grows with it too (exact storage is bounded by the
code width). Measure retention with and without separation under the same observed keys;
see the [separation tests](../tests/test_separation.py). Biological expansion and sparse
coding motivate the construction, but do not establish equivalence to hippocampal learning.

`SynapticMemory` supports the same expanded write/read coordinates, but requires
`separator.center=0`: moving the separator's mean would move the address of persistent
records. If centering is needed, apply one fixed training-fitted transform to both inputs
and queries. This protects a coordinate convention; it does not prevent interference
between overlapping keys or representation drift in a separately trained encoder.
