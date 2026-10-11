# RecordPatchNet

`RecordPatchNet` is a temporal patch whose context is linear and gated and
whose memory of particular readings is a record store inside the patch. The
nonlinearity sits at the ports. Pin a release or a commit when reproducing an
experiment.

For inputs `u[t]`, context `h[t]` and outputs `y[t]`:

```text
l[t] = sigmoid(g + G u[t])               per-channel retention in (0, 1)
z[t] = tanh(B u[t] + b)                  the input port
h[t] = l[t] * h[t-1] + (1 - l[t]) * z[t]
m[t] = read(code([u[t] * sqrt(n) / s, r * h[t]]))   the record read, a port value
y[t] = C h[t] + c + m[t]
```

The context is a convex combination, so a bounded initial context stays
bounded. With inputs and parameters fixed, the seam residual
`h[t] - l[t] h[t-1] - (1 - l[t]) z[t]` is linear in the context path. With the
linear readout its energy is quadratic, and the free path is its unique
zero-defect normal form. The two teaching-detuned paths are unique when their
quadratic systems remain positive definite; the solver's convergence result
must be checked. Categorical ports have a different loss, described below.
Channels start at retention timescales log-spaced from two moments to
`slowest` moments. The gate parameters are learned: `slowest` is an
initialization scale, not a hard memory horizon. `r` reads each channel in the
unit of its initial fluctuations.

The path can change at every moment while satisfying its temporal equations.
That is different from repeatedly revising an interpretation of one fixed
observation. The [equilibrium and world-model guide](equilibrium-world-models.md)
distinguishes environment time, inference iterations and learning, and states
which parts of a shared world-model architecture remain unimplemented.

## One observed path

```python
import numpy as np
from cadence import RecordPatchNet

net = RecordPatchNet(inputs=17, hidden=8, outputs=6, seed=2, cells=4096, active=32)
rng = np.random.default_rng(7)
inputs = np.zeros((1, 16, 17))
inputs[0, 0, 0] = 1.0                            # one cue
inputs[0, np.arange(16), 1 + np.arange(16)] = 1.0  # sixteen distinct heard events
identities = rng.integers(6, size=16)
target = np.eye(6)[identities][None]

first = net.observe(inputs, target, rate=1.0, backtrack=True)
assert first.updated and first.writes == 16
recalled = net.imagine(inputs, state=np.zeros((1, 8)))
assert np.array_equal(recalled.output[0].argmax(axis=1), identities)
```

`observe` predicts the path with the records as they stood when the call
began, moves the slow parameters against the adjoint gradient of the
precision-weighted half mean squared error of the slow readout `C h + c`,
and then writes what that readout got wrong at each reading into the
reading's records. The record read never enters the slow gradient: the slow
parameters learn the observation itself, and the record patches their
current error until they have. With `backtrack=True` a parameter step is
admitted only after a target-free replay from the original boundary lowers
the slow readout's loss, trying up to sixteen halved rates. `write=False`
learns without writing. The final free context becomes the live state;
`advance` carries context without learning or writing, `imagine` is
private, and `reset` clears context while keeping parameters and records.

## The reading: two blocks of unit variance per unit

The record's key is the input block and the context block side by side. The
context is read in units of each channel's fluctuation (`r`), which gives its
units unit variance. The input block is scaled by `sqrt(n) / s`, where `n` is
the number of input ports and `s` the running rms norm of witnessed inputs, so
its units have unit variance too and a cell's drive has unit scale, with the
fixed offset a small preference. Without that scaling the context block
outweighs the input block, the address barely moves with the event heard, a few
hundred hub cells take most of the activations, and the store forgets its
corpus behind its last writers.

Two further options are off by default. `record_averaging` makes a cell's write
rate one over its written mass with `record_rate` as the floor, so a fresh cell
takes its first outcome whole and a familiar one averages: it calibrates reads
at unfamiliar readings and improves retention, and slows adaptation to a new
stream. `record_homeostasis` moves each cell's offset toward an equal activation
share, which spreads the code further and costs adaptation as well.

## Records hold what the slow model does not know

Each write target is `y_obs[t] - C h[t] - c`. A reading the slow parameters
have learned leaves a record near zero, so the store holds only the
residual the slow model has not absorbed. One write corrects its own
reading; readings whose codes overlap move by their overlap, as in the
record theorems. Keys of neighbouring moments share the slowly varying
context, so one write is not exact and the delta rule converges over
repeated observation. Readings that differ only by the bar they belong to,
under a periodic clock, are separated only partly by a leaky context; a
heard stream, in which the previous event is an input, supplies the rest.
Whether carried context alone can return a theme after many bars is an
open measurement, not a property of this class.

Records are read with the tables at the start of the call and written after
the path, so a prediction is a function of the parameters, the records and
the boundary alone, and a private branch reproduces it exactly. To let a
record written at one moment inform the next, observe in shorter paths.

## Diagnosing a record store

A record store can fail silently: prediction with records still beats
prediction without them, because the store always holds something. Four numbers
show whether it holds the right thing. Replay the readings of a training stream
through `records.code(..., valued=False)` and count, per cell, how often it is
active.

- **Cells in use** and **the share of activations taken by the most active
  cells.** With `active` of `cells` winners the ideal share of any 200 cells is
  `200 / cells`. A store using a quarter of its cells, with most activations in
  the top 200, is addressing badly.
- **Code overlap when only one block of the reading changes.** Hold the context,
  change the input, then the reverse, and compare the two codes. An address that
  does not move with a block cannot store anything about it. A well-scaled
  reading moves substantially with the input and keeps most of its cells when
  only the distant context moves.
- **Held-out scores with records frozen, zeroed, writing online, and writing
  online from empty.** Frozen far below online means the store forgets its
  training stream; online equal to online-from-empty means the trained records
  contribute nothing to a new stream.
- **The read at readings no stream has written,** for instance a rollout from
  silence. A read far from zero there is interference from unrelated writers,
  not memory.

## What a write rate stores

The delta rule at a rate near one half makes a cell hold its last few writers.
That is the right memory for a new stream: its first moments overwrite the cells
they touch, and the read then calibrates the prediction to that stream. It is
the wrong memory for a corpus: with records frozen such a store recalls little
of its training material. `record_averaging` makes each cell average its writers
instead, with `record_rate` as the floor; retention and calibration at
unfamiliar readings improve, adaptation to a new stream slows in proportion. One
table cannot do both at full strength; choose by which the task reads.

## Recall is by content, not by position

A record is keyed by the reading: what was heard and the context. Where the
stream recurs, the key recurs, and the codes at the same place one loop apart
share most of their cells. When the stream drifts in phase, so that the same
place in the bar hears another event, the codes barely overlap and nothing
written there is read back. A patch recalls a moment when it meets the same
event in a similar context, not when it reaches the same position. If a task
needs recall by position — the same bar of a phrase, the same step of an
episode — position has to be in the reading as an input of its own, with enough
ports to move the code; a clock of eight ports among eighty inputs does not.

## Boundary readings

A reading met once per stream, such as the wake moment with nothing heard,
receives a handful of updates per epoch. The slow parameters barely learn it and
its record cells are shared with common readings, so the first prediction of a
rollout from silence is poorly determined. Make the boundary a common reading by
convention instead of hoping it is learned: let the wake hear a count-in, the
last event of the loop, so the rule the patch knows best produces the first
event.

## Detuning as the acceptance check

```python
net.reset()
adjoint = net.observe(inputs, target, rate=0.0, write=False).delta
net.reset()
contrast = net.detune(inputs, target, beta=1e-4)
assert contrast.converged
for name, value in adjoint.items():
    assert np.linalg.norm(contrast.contrast[name] - value) < 1e-6 * np.linalg.norm(value)
```

`detune` solves the two detuned equilibria of the quadratic energy by
conjugate gradients from the free path and returns the centered contrast of
the energy's parameter derivatives. The energy describes the slow patch: its
readout residual is `y - C h - c`, and the record read lies outside it. For
this patch the contrast equals the adjoint gradient up to terms of order
`beta^2`; `observe` computes the same learning signal by one backward scan.
Neither route sees the record: the read is a port value added to the
readout.

## Cost and state

One update is one forward scan, one backward scan, one record read per
moment and one write per observed moment. Work per moment is linear in the
context width plus the record projection; state is the context, the
parameters and the record tables, constant in stream length. The dense
hidden-width block messages of `TemporalPatchNet` do not appear.

## Checkpoints

`snapshot`, `restore`, `save` and `load` carry the parameters, the record
tables, the running mean and counts, the output precision and the live
context. `readback` exposes the detached state, update and write counts, the
parameter revision the state was computed under and the record entry count.

These snapshots contain the patch's continuation state, not the application's
observation history or a planner's external state. An application that branches
imagination also owns its random generator, action adapter, observation
normalization and any additional memories. Freeze those with the model during
a decision, and verify that private reads leave the factual snapshot unchanged.
Keep imagined targets distinguishable from observed outcomes; `sleep` consumes
model-generated targets and does not authenticate them as observations.

## Running a trained patch outside Python

The forward pass needs no library. Export the six parameter arrays, the
record table, the running mean of the reading, the input norm and the
record configuration; everything else is regenerated. The record projection
and offsets come from the seed: the `Mulberry32` generator, then
`normals(n)` by Box-Muller over `2 * ceil(n / 2)` draws with the radii from
the first half of the draws and the angles from the second half, cosines
before sines; the projection takes `reading * cells` normals divided by
`sqrt(reading)`, row by row over the reading, and the offsets take the next
`cells` normals times `bias`. A moment is then the gate and the context
update, the reading `[u * sqrt(n) / s, r * h]` minus the mean, the drives,
the `active` largest of them (a heap of that size is enough), their
positive parts normalised to unit length, the table rows weighted by that
code, and the readout. At 208 reading units and 8,192 cells a moment is about 1.7 million
multiply-adds, which a port in another language runs in milliseconds. Ship the
table as float32 and keep an archived Python rollout as the port's parity test.

## Categorical ports

Symbols (words, moves, note names) are taught as distributions, not as levels.
With `groups` the slow readout ends in one softmax per group of ports and
learns by cross-entropy, whose output error is `softmax - target`:

```python
import numpy as np
from cadence import RecordPatchNet

net = RecordPatchNet(inputs=17, hidden=8, outputs=6, seed=2, cells=4096, active=32,
                     record_rate=1.0, groups=(6,))
inputs = np.zeros((1, 16, 17))
inputs[0, 0, 0] = 1.0
inputs[0, np.arange(16), 1 + np.arange(16)] = 1.0
identities = np.random.default_rng(7).integers(6, size=16)
first = net.observe(inputs, np.eye(6)[identities][None], rate=0.0)
assert np.allclose(first.prediction.slow_output.sum(axis=-1), 1.0)
recalled = net.imagine(inputs, state=np.zeros((1, 8)))
assert np.array_equal(recalled.output[0].argmax(axis=1), identities)
```

A record then holds `onehot - softmax(C h + c)`, a residual bounded by one in
every port, so the record algebra is the one above: a reading the slow weights
have learned leaves a record near zero. The prediction `softmax + read` can
contain negative entries and is not generally a distribution. Its largest
port can be used for classification, but sampling, probabilistic planning and
log-loss evaluation require a declared nonnegative, normalized distribution
for each group.

The current loss implementation clips each entry to at least `1e-12` and then
normalizes the group. A residual that pushes the true category below zero can
therefore incur a very large loss. That exposes a problem with the resulting
probabilities even when the winning category is correct on many other rows.
Report mean held-out log loss and calibration for the declared normalization,
alongside accuracy and the slow-only scores. State the clipping floor; a median
loss may supplement these measures but must not replace the mean or conceal
rare confident errors. A learned probabilistic combination of records and
slow logits would be a different implementation and needs its own tests.

Each group's teaching weight is the mean `output_precision` of its ports.
`detune` refuses categorical ports: the quadratic detuning result applies only
to the linear readout. Default nets keep checkpoint format 2; a net with
`groups` or batch writes saves format 3.

## Writing a batch

`record_writes="batch"` writes all moments of a call at once through
`Records.write_batch`: every error is taken against the tables as they stood,
and each cell moves by the mean of the moves its writers would have made alone.
One writer reproduces `Records.write` to rounding; writers that agree move a
shared cell as far as one of them would, so a batch does not overshoot. Within a
call, sequential writes let a later moment overwrite an earlier one that shares
its cells and so need more presentations to settle a repeated path; the batch
average does not.

## Do not record a choice

When a reading has several right outcomes (a message that can be said in four
ways), the slow readout learns their distribution and its error at that moment
is the choice that happened to be taught. A record of it pulls later
predictions toward one arbitrary choice and spends cells. Write records for
what was observed once and must be reproduced (a sentence read, a phrase
heard), and leave choice points to the slow weights.

## Sizing a store for recitation

A store recites a sequence exactly only when it has more cells than
associations to hold, and it needs several presentations, because keys of
neighbouring moments overlap. Closed-loop recitation multiplies the per-word
rate over the sentence, so exact recitation needs the per-word rate near one:
about seven cells per association, tens of presentations, and slow weights that
have learned most of the material first. Teacher-forced word accuracy is
reached much earlier than exact closed-loop recitation; measure both.

## A store narrower than its port

A port of thousands of categories need not give the store one column per
category. With `record_width=w` the cells hold a fixed random sign code of the
residual, `residual @ R` with `R` of shape `(outputs, w)` and entries
`+-1/sqrt(w)`, and the read is decoded by the transpose, `held @ R.T`. A stored
residual comes back with crosstalk of standard deviation
`|residual| / sqrt(w)` per port, which the largest port survives. The delta rule
has to compare like with like: the coded residual with what the cells hold.
Taking the error after decoding and projecting it again multiplies the step by
`outputs / w` and the store diverges. A code of a few hundred columns recites
about as well as one column per category at a fraction of the memory.

## A grammar from records alone

Records generalise by overlap, and that is enough for a small grammar. Give a
speaker a message (act, kind, number and agreement features) and the word it has
just said, leave the slow weights at their random initial values, and write one
delta-rule pass over a corpus of (message, sentence) pairs. Asked for messages
it never met, the store alone produces sentences inside the grammar for most of
them and prefers the grammatical member of a minimal pair (a/an, is/are,
deal/deals, both/all). A new message shares most of its features with taught
ones, so its reading touches their records and the read averages them: the
record principle doing agreement and word order without a gradient. Scores are
greedy decoding; sampling the opening words collapses them. A store of that size
is not capacity-matched against a slow learner, and the grammar is finite, so
this says what one pass of writes buys, not that the store beats a trained
model. The held-out messages are new combinations of taught features, not new
vocabulary.

The rule the settings follow: address with enough active cells that neighbours
overlap, write at rate 1 when each reading is written once, and keep the token,
the cue and the context in balance in the address. Too few active cells lose the
generalisation; a write rate below one in a store written once leaves each record
holding part of its residual; more passes over the same material lower the valid
share rather than raising it; the output code's width barely matters.

## Acquisition in two phases: records by day, weights by night

A gradient learner acquires a corpus one way: a small move of every weight at
every token, for as many passes as it takes. A record patch acquires in two
phases, and the phases are the two learning rules the patch already has.

**By day, what is observed is written, once.** A reading is the message or cue,
the event just heard and the context; its record takes the outcome in one write
and holds it exactly (Theorem: one-shot memory). The slow weights do not move.
This is why the patch can be told a fact and use it in the next sentence, why it
can take a corpus in one pass, and why nothing it learns by day disturbs what
its weights hold. The cost is that a store generalises by overlap and no
further, and that it disturbs itself when it is shared: new material written
into a store that also holds a library degrades the library a little. The store
is linear, so two stores read together are one store; what keeps a conversation
apart from the library is which store a cue reads. A store of its own protects
the library, and does not stop new material from overlapping itself.

**By night, the slow weights take what the store holds, from dreams.**
`sleep(cues)` dreams every cue of the day once (the free path with the records:
what the patch would answer awake), teaches the slow weights those fixed dreams
by the ordinary contrast with `write=False`, and at dawn writes the dreams back
so the store holds only what the weights did not take. Nothing outside the patch
is consulted. Dreaming is necessary rather than decorative: the slow weights
need many presentations of a regularity, and once the corpus is gone the store is
the only place those presentations can come from. It is also where
generalisation is made: the completion of a cue the store never met is what the
records of its neighbours agree on, and the slow weights learn that agreement as
a rule, so after a night the weights alone can answer cues the store only
guessed at.

**What sleep does not do.** Dreams carry the store's errors as faithfully as its
regularities: where the store was wrong at bedtime, the night changes nothing,
while a teacher with the corpus present corrects it. Dreams from recombined cues,
which land on combinations never taught, teach the store's guesses as facts;
dreams from the cues the store actually met let the smooth weights extend the
pattern. Sleep restructures what is held and does not correct it. Correction
needs a critic, a re-reading of the disputed cue, or another day. A night also
does not beat a teacher that still has the corpus, and the gap grows with the
size of the corpus.

**Why the dreams are fixed and the store rewritten.** A record is the residual
against the slow readout at the time it was written. Dreaming again after every
update would teach the weights their own drift, and reading the old residuals
against new weights would count each outcome twice. So the night fixes its
dreams at bedtime, and dawn re-references the store: two passes of writes leave
it holding only what the weights did not take, and a new fact written the next
day lands beside them. The store is fast, exact and local; the weights are slow,
smooth and general; the night moves knowledge from the first to the second.

**When to sleep.** The night pays when the observations are gone and the store
is the only copy of them. A learner with the world still present learns by day
at a slow rate above zero: it has fresh observations at every moment, where a
sleeper trades its actions for second-hand presentations of what its store
holds. What the store reads at bedtime bounds the night, so a store that is
drowned by the day (`record_averaging` for a store written hundreds of times per
cell) teaches a drowned rule.

## Maps: a structured input port

A record patch reads its inputs through the port `B u` (and the gate through `G u`).
By default that is one dense row per context channel. `StructuredPort` (in
`cadence.ports`) reads a declared layout instead: a `MapBlock` is a tied local
kernel over a `(channels, height, width)` grid of the inputs, the same kernel at
every position, whose output is a grid of channels laid out as retinotopic maps;
a `DenseBlock` is a plain matrix over a slice of the inputs. The patch's
equations do not change. The port supplies the three linear maps the adjoint
scan needs (apply, transpose, parameter gradient), so learning is the same rule,
and the context width is the port's output count.

```python
import numpy as np
from cadence import RecordPatchNet
from cadence.ports import DenseBlock, MapBlock, StructuredPort

fovea = MapBlock(start=0, channels_in=3, height=24, width=24, channels_out=8, kernel=5, stride=2)
rest = DenseBlock(start=fovea.inputs, inputs=52, outputs=64)
port = StructuredPort(fovea.inputs + 52, [fovea, rest])
net = RecordPatchNet(port.inputs, port.outputs, 4, port=port, cells=4096, active=32)
```

A port of dense blocks covering the inputs is the plain patch (the tests check the
outputs and the updates agree), and a map block answers the same thing wherever it
appears (a thing at one place and the same thing two positions on give the same
channel response two cells on). The blocks are genes: which slice, what grid, how
many channels, what kernel, what stride. `mask=` names which inputs the port hears,
one flag per input: a masked input is zero to every block in the forward map, its
transpose and the gradient, so which channels a patch reads is a gene too, without
touching the weights (a steering patch that reads the surprises and not the probes,
or a brain deaf to the channel a tested mechanism reads). A structured
port's checkpoint carries the port's layout and its mask; its `B` and `G` are the
blocks' kernels, packed.

## Two patches in depth

The gate of a record patch sees only the present input. That is enough for a
conjunction of the last two symbols, which the slow weights learn alone, because
the gate multiplies the carried context. It is not enough to tell a subject from
a later noun: on subject-verb agreement across prepositional phrases with nouns
of the opposite number, one patch holds up with one distractor and fails with
two. A second patch that reads the first patch's context has a gate that sees
what came before, and holds up throughout:

```python
import numpy as np
from cadence import RecordPatchStack

stack = RecordPatchStack(inputs=5, hidden=8, outputs=4, seed=5, cells=2048, active=16,
                         groups=(4,), record_rate=1.0)
symbols = np.random.default_rng(4).integers(5, size=(3, 8))
target = np.eye(4)[(symbols + np.roll(symbols, 1, axis=1)) % 4]
seen = stack.observe(np.eye(5)[symbols], target, rate=2.0, backtrack=True)
assert seen.updated and seen.writes == 24
```

`RecordPatchStack` is a context patch below a complete `RecordPatchNet` that
reads `[u, r1 * h1]`; the upper patch owns the readout and the records. The
upper adjoint scan hands its input gradient to the lower scan, and the
gradient matches finite differences in both patches. Work per moment stays
linear in the two widths.

For a single patch the previous-state Jacobian is diagonal when its input is
fixed. Its gates still multiply carried history, so this is not a theorem that
the patch represents only one level of conjunctions, and in a stack the upper
nonlinearity mixes the lower context. The agreement result supports that
composition on the measured task; it does not establish a general world model.

Each component's quadratic seam statement holds conditional on its inputs.
Freeing both layers together makes the upper nonlinear port depend on the
lower state, so a jointly coupled energy is not automatically quadratic.
`RecordPatchStack` supplies an adjoint learning scan, not a joint `detune`
solver or a joint equilibrium convergence guarantee.

## Several patches joined by ports

`JointRecordPatches` couples several record patches through a fixed number of repair rounds. A
`Port(source, target, start, width)` declares a seam: the band
`[start, start + width)` of the source patch's scaled context `r * h` (the unit
its records read it in, and the unit the stack hands upward) is an input of the
target patch in the same moment. The target's gate and input port read it
through their ordinary weights, and its records key on what crosses; nothing
else crosses. The joint energy adds `1/2 |p[t] - S r h_source[t]|^2` per port to
the patches' seam energies, and at every moment `rounds` Jacobi rounds from the
previous moment's contexts approximate the coupled equations: round one is the delayed
port, each further round re-reads the other patch's context of this moment. For
a single one-way port from an independent source, two undamped rounds reach the
fixed point; longer port chains need more rounds. Recurrent ports have no general
convergence guarantee here. Inspect the remaining equations independently before
claiming a joint equilibrium: a fixed round count is not a stopping certificate.
The seam residual per moment and round is
the fourth instrument beside the residual, the surprise and the update.

Snapshots retain whether ports were cut, alongside each cortex's learned and
live state. Invalid/nonfinite paths cannot become live contexts. Joint record
writes are transactional: failure on a later cortex restores earlier writes.

```python
import numpy as np
from cadence import JointRecordPatches, Port, RecordPatchNet

eye = RecordPatchNet(inputs=5 + 3, hidden=8, outputs=4, seed=1, cells=256, active=8, groups=(4,))
ear = RecordPatchNet(inputs=5 + 2, hidden=6, outputs=4, seed=2, cells=256, active=8, groups=(4,))
brain = JointRecordPatches([eye, ear], own_inputs=[5, 5],
                           ports=[Port(1, 0, 0, 3), Port(0, 1, 2, 2)], rounds=2)
rng = np.random.default_rng(0)
xs = [np.eye(5)[rng.integers(5, size=(3, 8))] for _ in range(2)]
ys = [np.eye(4)[rng.integers(4, size=(3, 8))] for _ in range(2)]
seen = brain.observe(xs, ys, rate=2.0, backtrack=True)
assert seen.settled.seam.shape == (3, 8, 2, 2)      # batch, time, rounds, ports
```

Learning is one backward scan through the moments and the rounds; with
`cross_adjoint=True` the gradient of a target's loss reaches the source's context
through the port, with `False` the port is a plain input. The step is admitted by
the library's backtracking on the joint slow loss. Records are read at the
settled readings after the scan and written after the path, as in one patch.
Snapshots are the patches' snapshots plus the topology; `imagine` changes
nothing; `cut = True` makes every port carry zeros, the ablation. Which patch
hears which, the band and the width are a genome for `evolve` with `genes`;
the channels are ordered by timescale, so the band decides what crosses: the
fastest channels carry the source's current input, the slowest its carried
state. Measure a port by cutting it: a joint port that lifted a dependent stream
should drop it again when `cut=True`. There is no joint `detune`, because the
joint energy is not quadratic in the joint path.

## What this class does not do

Planning through action ports, protected responses through `TemporalMemory`
and coupled components beyond the two-patch stack are not yet available for
this class. The stack has no `detune`. The record
rate, the number of active cells and the reading scale are supplied. There
is no automatic importance, forgetting, or learned relevance: a record is
written for every observed moment and moves only when its cells are read
again.
