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

## From pixels to keys

An observation reaches two things: the cortex, which settles on it, and the memories, which
key on it. A sense organ's own code, a retina, a cochlea, a V1 region, belongs to the
application or to the anatomy that builds the brain; `Brain.build((height, width), actions)`
and `Brain.compose((height, width), actions)` give the cortex the pixels through a
`visual_cortex`, feature maps with local receptive fields that learning shapes. The
library's generic compression is the brain's sparse code. `brain.observation_code(x)` is
what both memories key on: the observation itself, or with `code=units` the brain's own
code of it, `units` code neurons with positive weights that read each input through two
rectified channels, ON `max(x, 0)` and OFF `max(-x, 0)`, and every channel at least twice,
the largest sums of one unit in ten kept (102 of 1,024) and the code scaled to unit length
(a `PatternSeparator` with a fan-in and `cover=2`, owned by the brain and saved with it).
The code deals each channel two readers first, then fills each unit's fan-in of 32
channels at random from the seed (more channels per unit where 32 cannot read every
channel twice: 51 for 1,024 units and 401 for 128 units at 12,824 inputs). A single input
of either sign reaches the code, a signed observation reaches it in both signs, and a
nonnegative observation leaves the OFF channels silent. Too sparse a code leaves the
cortex and the memory too little overlap between a question and the moment it asks for.
The code compresses: channels read by the same units are told apart by their weights
alone, and an observation lit widely keeps one unit in ten, so the brain on its raw
observation is the control a code is chosen against. Left unset, every brain keys on its
observation, whatever its width; a code of 1,024 units is the recommended setting above
1,024 inputs.

```text
observation ─► sensory region ─► association cortex ─► motor readout
      └──────► sparse code (fixed: 1,024 units, ON and OFF channels each read twice, one in ten active, unit length)
                   ├─► episodic memory: place records, and context records with the command
                   └─► recall memory: the slots and their address
```

- **What keys on it.** The episodic memory's place records key on the code and its context
  records on the code with the issued command; the recall memory's slots and address key on
  it. A coded brain's stores take no separator of their own: the code is the separation. An
  uncoded key (the observation itself) keeps the stores' separators, dense by default
  (`memory_fan_in` 0); above 256 key values the recommended setting is sparse, each of the
  1,024 cells reading 32 values (`memory_fan_in` 32), with 24 cells active as before.
- **The cortex's input.** `Brain.compose(inputs, actions, code=1024)` makes the sensory
  region the code: every observation is coded before the drive, so 12,824 inputs reach the
  cortex as 1,024 units through 1,024 x 64 sensory synapses in place of 12,824 x 64, and the
  memories key on the same code. The founder leaves the cortex and the memories on the
  observation; `code=units` is opt-in.
- **Custom anatomies.** `Brain(connectome, ..., code=units)` keys the memories on the code
  with the cortex on the observation, and names a coded sensory region with `inputs=` (the
  observation width) beside `code=` (the region's width). A recall memory, opt-in there,
  needs `recall` and `next` regions of the code's width (`Brain.observation_width(inputs,
  code)`); the heard history holds blocks of the sensory region's width.

At 12,824 inputs (a 128 by 100 camera and 24 body senses, an association region of 64
neurons, three actions), the episodic memory's separators and the recall memory's address
map, keyed on the observation itself (the founder at every width) and on the recommended
code of 1,024 units:

| Part | On the observation (the founder) | On the code (`code=1024`) |
| --- | --- | --- |
| Key width | 12,824 values | 1,024 units |
| Episodic separators, place and context | 105.1 MB and 105.1 MB, dense (0.63 MB each with `memory_fan_in=32`) | none: the code is the separation |
| The code's tables, indices and weights | | 0.84 MB |
| Recall address map `W_x`, `W_c`, `b` | 1,322 MB (`W_x` 1,316 MB) | 8.9 MB |
| Together | 1,532 MB | 9.8 MB |
| Sensory synapses into a module of 64 | 820,736 | 65,536 |

The recall memory with cortex values (`recall_value="cortex"`) and an association region of
512 neurons holds 14.5 MB at that width, [below](#recall-the-streams-own-moments).

<a id="three-kinds-of-memory-in-brain"></a>

## Memory in the composed brain

| Mechanism | What it retains | What changes it | What survives `brain.reset()` |
| --- | --- | --- | --- |
| Working `Trace` | Recent association-region activity used as context | Each admitted live action | Nothing; the trace is cleared |
| `Efference` copy (opt-in gene) | The fading one-hot of the commands the stream issued | Each admitted live action | Nothing; the copy is cleared |
| `Recall` memory (opt-in gene) | The stream's recent moments as slots, each addressed before its settle ([below](#recall-the-streams-own-moments)) | Each admitted live action and each moment `wait` senses (the aroused ones alone with `recall_write="aroused"`); the address map by each teacher lesson, and with `recall_address_reward` by reward | The address map; the slots are emptied |
| Graph plasticity | Learned synaptic parameters and biases | Supervised contrasts and reward-modulated eligibility | Learned parameters |
| `SynapticMemory` | Cue-to-outcome associations, with fast per-stream `F` and shared persistent `C` | Observed chosen-action outcomes | Both `F` and `C` |
| Episodic memory (`memory_choice=True`) | The same store keyed by the observation, and a second keyed by the temporal context; their mean votes in the choice | Known outcomes, demonstrations, replay of the path to a reward | The records; the temporal context and replay path are cleared |

These are functional software mechanisms. Their capacities follow their state
dimensions, coding and update rules; a human working-memory estimate does not set
the number of durable associations they can retain. A lost response can come
from a missing trace, interfering parameter updates or interfering record writes.
Test the mechanism that was used to acquire the response.

`predict` and `accuracy` ignore the working trace, the efference copy, the recall
memory and the associative store.
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

With `memory_choice=True` the composed brain's hippocampus is an
[episodic memory](episodic.md): keyed by the observation and, in a second store, by the
temporal context, it records known outcomes and demonstrated actions
(`memory_demonstrations`) and its records vote in the choice, added to the motor readout
before the softmax. By default it is the earlier store: `teacher=` teaches
graph parameters without inserting its label, the reward loop stores the actual chosen
action's observed reward, and the read is a linear drive into the graph whose resulting
action must still qualify the complete neural equations. Direct `observe` calls are an
explicit key/value acquisition rule; they test that store rather than graph plasticity.

Measure acquisition first, then expose the same brain to competing experience
and query old and new responses without teachers. Rehearsal can protect recall
and must be counted as additional teaching or record writes. Finite storage and
successful bounded recall do not establish lifelong retention.

Working-trace amplitude, retention and scale are task-dependent settings. Since
0.82.0 the composed brain carries exactly its last interpretation (retention 0),
holds it clearly in the prefrontal region (amplitude 3.0) and reads it back through
plastic synapses that start quiet (`working_memory_scale` 1.2, where earlier releases
used 12); [context](context.md) explains it. A stronger trace retains earlier context
and can also interfere with acquiring a new relation, and no single setting is a
general fix. Where the previous moment carries nothing the decision needs, the trace
is interference the actor has to learn away: on a continuing contextual bandit the
composition of earlier releases held the hit rate at chance over several seeds while
amplitude 0.0 reached 1.0, and a small amplitude with a smaller actor rate also
reached 1.0
([night-replay chamber](../benchmarks/replay/README.md#the-working-trace-on-a-continuing-bandit)).
The [vanished-cue instrument](../benchmarks/recall/README.md) measures a
continuing brain against checkpoint-matched trace interventions and an
explicitly trained history comparator; its software checks validate the
instrument, not a recall horizon. For demonstrations without a reward protocol,
use the [supervised-only stream](continuous.md#a-supervised-only-stream).

## Recall the stream's own moments

The working trace keeps what the brain made of the moment before, and the
[heard history](context.md#hear-the-past-at-several-timescales) keeps the observations at a
few fixed timescales. Neither gives back one earlier moment when the present asks for it.
The recall memory, `Brain.compose(..., recall=True)`, keeps each stream's recent moments as
slots, addresses them before each settle and holds what it recalled through the settle: a
held-input recall. Its two regions add one neuron per observation
coordinate each: `compose(250, 2)` grows from 380 to 880 neurons, which a world that charges
per neuron feels; a brain's `code` bounds the regions at the code's width.

```text
moment t:  observation x -> address q -> weights over the slots -> recall and next regions -> settle -> act
                                                                                                         |
           after the act: a new slot (key heard * x, value x); the slot before learns x as its next  <---+
```

- **One slot per moment.** After each act, and after each moment `wait` senses, every stream
  writes one slot: its key `heard * x` (`recall_heard` 1.0, `x` the observation as heard),
  its value `x` (`recall_value="event"`) or the settled activity of the association region
  (`"cortex"`), and its next, the value of the moment that follows, written when that moment
  comes. `recall_capacity` (64) bounds a stream's slots; the oldest is dropped first.
- **The address comes before the settle.** Before each settle the moment's address is
  `q = heard * x + context * h + x W_x + c W_c + b`, with `h` the fading history of the
  earlier events (`recall_context` 0.0, kept at `recall_context_decay` 0.5 per moment), `c`
  the settled association activity of the moment before and `W_x`, `W_c`, `b` a learned
  map that starts at zero. The slots are weighed once, by
  `p = softmax(q . K / 0.25 - 0.25 * age)` (`recall_temperature`, `recall_recency`), which
  `brain.recall.attended` keeps. With the map at zero the address is the event heard: a
  name finds the moment it was stated in before any lesson.
- **Two source regions, held through the settle.** The `recall` region is driven at
  `3.0 * sum p V` and the `next` region at `3.0 * sum p N` (`recall_amplitude`;
  `recall_next=False` leaves the next region out), a drive set before the settle and held
  through it as a sensory drive is: the settle answers on the mixture the address chose and
  does not revise the choice. Both regions reach the association region through one-way
  plastic synapses of initial magnitude `recall_scale` (1.2), which the settle's contrast
  teaches as it teaches the sensory ones. A moment's address cannot depend on what that
  moment recalls: a second recall takes a second moment.
- **The address lesson.** After each teacher lesson (`step(..., teacher=...)` or
  `Brain.lesson`) a rule of its own moves the map. It reads the movement of the association
  region between the nudged phases back through the recall and next synapses, scores each
  slot's value and next against that movement (the stored values themselves, not their
  effect through the source neurons), weighs the scores with the address's own attention
  (`recall_address_temperature` 0.1), and takes a least-mean-squares step of the map
  (`recall_address` 0.01) toward the keys so weighted, at the rows the lesson taught. Each
  row's rate is bounded by the row's own energy, `min(0.01, 1 / |z|^2)` on the map's inputs
  `z = [x, c, 1]`, so a row taught alone moves its address at most onto its target. The
  rows' steps are averaged, and a row's step moves every row whose inputs overlap its own,
  so in a batch one row can pass its target; the batch error, each row's squared distance
  to its target weighted by the row's rate, does not grow under the step. A step whose
  result is not finite is refused and counted (`address_refused`). The keys hold the fixed
  part of the address (the event heard, and with `recall_context` the history before it),
  so a target never contains the map's own output. At the founder
  `recall_address_reward=0` rewards through `learn` and `live` leave the map as it is.
- **Streams and episodes.** Each stream has its own slots. `reset` empties them, an ended
  episode empties its stream's slots, and imagination reads the live slots and writes none.
  The address map is learned: a reset keeps it, `parameters()` counts it and a checkpoint
  saves it.
- **Wide inputs.** The memory keys on the [observation code](#from-pixels-to-keys): the
  observation itself, or with `code=units` the brain's sparse code of it. The keys, the
  event values and the address map follow that width; on the observation itself the address
  map is inputs by inputs, 1,322 MB at 12,824 values, and a code of 1,024 units is the
  recommended setting above 1,024 inputs. `recall_value="cortex"` stores the settled
  association activity in place of the event and sizes the two regions by the association
  width; the settled association activity does not carry a moment's content in a form the
  next settle can read back, so the founder stores events.

Options beside the founder change one part of the memory each and leave the founder as it
is. Each is a gene beside `recall=True` that `retune` can change in a life, and a saved brain
keeps it ([genes](api.md#recall-memory-genes)):

- **The address step toward the named slots** (`recall_address_step="toward"`). The founder
  step (`"lms"`) moves the address toward the named keys `K* = sum p* K`; where the keys are
  shorter than the address, that shortens it, the attention flattens and the slot the lesson
  names can lose weight. The toward step takes the error `K* - sum p K`, the named keys less
  the mixture the address attended to, so that to first order the named slots' logits rise
  against the attended ones, whatever the keys' norms. The rate, its bound and the refusal
  are the founder's.
- **A power-law age charge** (`recall_recency=None`). A slot's logit is lowered by
  `r * log(1 + age)`, `r = heard^2 / temperature / log(1 + capacity)`: the weight of an
  unmatched old slot falls as a power of its age, `(1 + age)^-r`, two equal matches at ages
  1 and 3 keep weights in the ratio `2^r`, and a full match of a unit-length key outweighs
  the age of every slot the capacity holds. `brain.recall.charges()` gives each slot's
  charge; a number keeps the linear charge `recency * age`.
- **Aroused writes** (`recall_write="aroused"`). A moment is written when its act was
  chosen aroused, which is the act's own flag: a sampled act (`step`, `act`, an aroused
  `live` moment) writes, a greedy act (`act(..., greedy=True)`, a calm `live` moment)
  writes nothing, and a moment `wait` senses is written while the action it awaits was
  sampled. The cortex of the moment before and the history move on at every moment. In a
  batch a moment written for some streams is a column every stream has, held by the written
  ones, and each stream's latest held slot learns its next from that stream's next written
  moment. `recall_capacity` then bounds the store's columns, so a stream that wrote fewer of
  them holds fewer slots than the capacity.
- **A null slot** (`recall_null`, a number). A slot of logit `null / temperature` joins the
  read's softmax and reads nothing, so a moment that matches no slot recalls little:
  `attended` and the drives read the slots' shares, which sum to one less the null slot's.
  The address lesson weighs the slots alone.
- **Reward trains the address map** (`recall_address_reward` above zero). At each sampled
  act, before its moment is written, the step the address lesson would take were the act's
  action a lesson's label is the map's eligibility. The actor keeps it in a trace of its own
  beside the synapses' (it decays by `gamma * lam` per outcome and per calm moment, and an
  ended stream forgets it), and each outcome steps the map by `recall_address_reward` times
  the dopamine times that trace, checked before the graph's update; the report names the
  step (`address_reward_step`). The eligibility is a lesson's step, taken at the lesson's
  bounded rate `min(recall_address, 1 / |z|^2)`, so at `recall_address_reward` 1.0 a unit of
  dopamine moves the map by one lesson's step, and `recall_address=0` leaves the map as it is
  under reward too.

```python
import numpy as np
from cadence import Brain

optioned = Brain.compose(
    inputs=4, actions=2, modules=(16,), seed=1, recall=True,
    recall_address_step="toward", recall_recency=None, recall_write="aroused",
    recall_null=0.0, recall_address_reward=0.1,
)
fact, other = np.array([[1.0, 0.0, 0.0, 1.0]]), np.array([[0.0, 1.0, 0.0, 0.0]])
optioned.act(fact)  # sampled: chosen aroused, the moment is written
optioned.act(other)  # written too
optioned.act(other, greedy=True)  # greedy: nothing is written
assert optioned.recall.slots == 2
optioned.stimulus(np.array([[0.0, 0.0, 1.0, 0.0]]))  # a moment that matches no slot
assert optioned.recall.attended.sum() < 1.0  # the null slot keeps the rest
optioned.act(np.array([[1.0, 0.0, 0.0, 0.0]]))  # sampled: the map's eligibility is kept
report = optioned.learn(np.array([1.0]), np.array([False]), fact)
assert report["address_reward_step"] > 0.0  # the outcome stepped the map on its trace
assert optioned.describe()["genes"]["recall_address_step"] == "toward"
```

At camera width, with 12,824 inputs, an association region of 512 neurons, 64 slots, the
brain's code of 1,024 units and cortex values, one stream:

| Part | Size |
| --- | --- |
| Slots of one stream (keys, values, nexts) | 1.05 MB |
| Address map (`W_x`, `W_c`, `b`) | 12.6 MB |
| Code tables | 0.84 MB |
| The memory in all | 14.5 MB |
| One dense `PatternSeparator` of the same inputs into 1,024 cells, for comparison | 105 MB |

```python
import numpy as np
from cadence import Brain

indexed = Brain.compose(inputs=4, actions=2, modules=(16,), seed=1, recall=True)
fact = np.array([[1.0, 0.0, 0.0, 1.0]])  # a name (0) stated with a value (3)
indexed.act(fact, greedy=True)
indexed.act(np.array([[0.0, 1.0, 0.0, 0.0]]), greedy=True)  # another moment
name = np.array([[1.0, 0.0, 0.0, 0.0]])
drive = indexed.stimulus(name)  # the name addresses the slots before its settle
recalled = drive[0, indexed.connectome.populations["recall"]]
assert recalled[3] > recalled[1]  # the moment that shares the name comes back
report = indexed.lesson(name, [1])  # the lesson, then the address lesson at the taught row
assert report["address_rows"] == 1.0
indexed.act(name, greedy=True)  # the moment is written
assert indexed.recall.slots == 3
indexed.reset()  # a new stream begins with no slots; the address map is kept
assert indexed.recall.slots == 0 and indexed.recall.w_x.any()
```

The founder `recall=False` builds the brain without the two regions, byte-identical to the
brain before the option, and is the control. A custom anatomy (`Brain(connectome, ...)`)
opts in with `recall=True` and its own `recall` and `next` regions, one neuron per value
coordinate each, the width of the observation code with event values or of the
association region with cortex values. The read and lesson genes can be retuned; the
capacity, the value, the next region and the scale are construction settings
([genes](api.md#recall-memory-genes)). A saved brain keeps its slots and its address map
(format `cadence-generic/8`; `/9` with the brain's code) and continues exactly. Erase a copy's slots before the
query (`brain.recall.reset(batch)`) to check that the memory does the work. The binding
instruments keep the reference implementation, `benchmarks/binding/index_read.py`, which the
option reproduces moment for moment (`tests/test_recall_memory.py`).

Limits: the address is a linear map of the event and of the cortex of the moment before,
chosen before the settle and held through it, so the settle cannot change which moments it
receives and its stopping check covers the graph on that drive, not the choice. An
address that depends on a recalled token and the question together reaches it only through
conjunctions held in the cortex of the moment before. The slots are a record within one
stream; nothing consolidates across streams except through the plastic synapses that read
them.

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
