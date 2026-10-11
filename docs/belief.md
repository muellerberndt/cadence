# The belief patch

`BeliefPatch` keeps a belief that a learned transition carries forward under the action
that was executed, lets the evidence of the moment repair that belief through a few
iterations of one nonlinear map, reads the record store inside those iterations, and
answers through a linear readout the store patches. It is the smallest patch in the
library that can imagine on its own: the record patch's context is computed from the input
of the moment, so without an input it has no next state; the belief patch's transition
needs only the belief and a declared action.

This API runs a fixed repair budget, not a solve admitted by a global
fixed-point threshold. Its slow learner differentiates through that finite
computation. See the [contract guide](contracts.md) before interpreting its
results as equilibrium or local-detuning evidence.

```text
p[t]    = g * z[t-1] + (1 - g) * tanh(T [z[t-1]; a[t-1]] + t_b),   g = sigmoid(G [z; a] + g_b)
e[t]    = tanh(port(o[t]) + e_b)
z(0)    = p[t]
m(k)    = read(code([e[t], z(k)]))
z(k+1)  = (1 - alpha) z(k) + alpha tanh(F [z(k); e[t]; p[t]; m(k); 1] + f_b)      k < K
y[t]    = C z(K) + c + decode(m(K))
```

The three clocks stay apart: the environment's moments `t`, the repair iterations `k`
within a moment with the evidence held fixed, and learning between chunks. An imagined
step is the transition alone, with the store read at the expectation and nothing observed.

## What each part is for

- **The transition** answers what follows under an action. Scene and action meet inside
  the repair map, so the effect of an action can depend on what is where; the guide's test
  is the mixed difference of the belief over an observation and an action, which is zero
  for a port whose blocks never meet and nonzero here.
- **The repair** assimilates evidence: the map reads the belief, the encoded observation,
  the expectation and the store's read together, and moves the belief by a damped step.
  `BeliefPath.step` is the last move per belief unit and `residual` its size.
  A small damped move is not an independent fixed-point certificate or a
  guarantee that a familiar observation is represented correctly.
- **The store** holds the residual of the slow readout at the code of the final reading,
  coded to `record_width` signs, written once per observed moment with `write=True`. Its
  read enters the repair and patches the readout. No gradient reaches the store; that is
  the record patch's rule and it stays. `observe` leaves the store unwritten
  unless asked, because on the measured lanes writing it paid nothing. The
  trade-off is the record patch's: a store that holds what the slow model does
  not know, against reads that habituate to stale residuals.
- **Imagination** (`imagine(actions)`) leaves parameters, records and live activity
  unchanged and consumes no observation; cost counters include its work.
  A continuation conditioned on recorded future
  inputs is a different measurement; this method cannot make one.

```python
import numpy as np
from cadence import BeliefPatch, DenseBlock, MapBlock, StructuredPort

# a small retina with the action broadcast into its map, and a dense block of fields
port = StructuredPort(1 * 8 * 8 + 4, [MapBlock(0, 1, 8, 8, 4, 3, 2), DenseBlock(64, 4, 8)], broadcast=(64, 4))
brain = BeliefPatch(port, actions=3, belief=16, outputs=6, iterations=2, cells=256, active=8, record_width=16, seed=0)
rng = np.random.default_rng(0)
observations = rng.random((2, 8, port.inputs))         # (N, T, inputs)
actions = rng.random((2, 8, 3))                         # (N, T, actions)
targets = rng.normal(size=(2, 8, 6))                    # (N, T, outputs)
brain.reset()
seen = brain.assimilate(observations, actions)          # the belief follows the evidence
ahead = brain.imagine(actions[:, :4])                   # (N, H, actions): private, from the live belief
learned = brain.observe(observations, actions, targets, rate=1.0, write=True)   # one backward scan through every iteration; the store takes the residuals
assert learned.updated and learned.writes == 16
```

## Learning

`observe` computes the adjoint of the precision-weighted half mean squared error through
every repair iteration and the transition over the chunk, treating each store read as
given, and moves the parameters by `rate` times that gradient. The adjoint matches finite
differences to 1e-5 on every parameter group (`tests/test_belief.py`). The loss is a mean
over time and outputs, so the useful rate scales with their product; a wide picture port
needs a rate in the thousands, or the torch backend with an ordinary optimiser.

`belief_torch.TorchBelief` is the same slow half as a torch module with autograd, for
batches of streams on a GPU; `export()` and `load()` move the parameters in the library's
packing, so a patch trained there continues in NumPy with its records. Its `forward` takes
the `gains` and the per-row `observed` mask of the sections below, since the forward is
shared, and a gains tensor that requires grad receives the gradient into the gains; a
weighted loss, an admitted step and an external output gradient are the caller's loss and
optimiser there, and the readback of a moment stays on the library's path.

### Training the transition: the imagination loss

A belief patch trained on the one-step read alone learns to lean on the next
frame: the repair takes what it needs from the evidence, the transition carries
less each epoch, and the open-loop imagination decays while the one-step read
improves, until it falls below plain persistence. That is teacher forcing, and
the rule that names the repair is that a rollout must predict every input it
consumes.

The imagination loss trains that. From random moments `t0` of each chunk, imagine `H`
steps under the recorded actions from the belief at `t0 - 1` with no observation, and
penalise the drift of the imagined output from the truth along with the one-step loss:

```text
L = L_one_step + w * mean over starts of  1/2 * mean( W * [cumsum_k(y_hat[t0+k] - y[t0+k]) ; fields]^2 )
```

where the cumulative sum runs over the retina outputs (the imagined retina against the
true one), the fields are compared step by step, and `W` is the same output weighting as
the one-step loss. The gradient flows through the imagined transitions into `T`, `G`, `C`
and back through the repair into the belief the rollout started from, so the belief is
trained to carry what the transition needs. Two starts per chunk, `H = 8` and `w = 0.3` are a working starting point on
pixels: the imagination then explains a useful share of the changed cells
several decisions ahead and beats persistence, while the one-step read holds up
against a convolutional recurrent model of the same size. Not every run takes —
a seed can learn the one-step read and no imagination at all, the transition
having settled on persistence while the repair does the work — so carry the
open-loop curve in every receipt, and raise the imagined term's weight first.
The recipe lives in the application, around the ordinary loss; the library
supplies the imagination with a `state` and no observation.

## A gain per block inside the repair

Each block of the observation port carries a gain that multiplies its encoded evidence
before the repair map and the store read see it, `e_b <- gain_b * tanh(E_b o_b + e_b)`, so
the gain acts in every repair iteration and in the store's code, on the equations of the
moment. A weighing applied after the outputs would be a reparameterization the readout
could learn around; a gain inside the repair changes what the belief is repaired toward.
`assimilate`, `observe` and `imagine` take `gains=`, one per block `(blocks,)`, per stream
`(batch, blocks)` or per moment `(batch, time, blocks)`; the adjoint returns the gradient
of the loss into each gain per moment as `gain_gradient`, and the path carries the gains
used. With `de` the gradient into the scaled evidence, the gain's gradient is the sum over
the block's units of `de * e_raw`, and the port's gradient uses `de * gain * (1 - e_raw^2)`.

```python
import numpy as np
from cadence import BeliefPatch, DenseBlock, StructuredPort

eye, ear = DenseBlock(0, 6, 8), DenseBlock(6, 2, 4)                # two senses, two blocks
cortex = BeliefPatch(StructuredPort(8, [eye, ear]), actions=1, belief=8, outputs=2, cells=128, active=4, record_width=8, seed=1)
rng = np.random.default_rng(1)
o, a, y = rng.random((4, 6, 8)), np.zeros((4, 6, 1)), rng.normal(size=(4, 6, 2))
cortex.reset()
cortex.observe(o, a, y, rate=1.0, write=False)                    # a readout, so the loss reaches the evidence
gains = np.tile([1.8, 0.2], (4, 6, 1))                            # (N, T, blocks): the eye weighs 1.8, the ear 0.2
cortex.reset()
taught = cortex.observe(o, a, y, rate=0.0, write=False, gains=gains)
assert taught.gain_gradient.shape == (4, 6, 2) and np.abs(taught.gain_gradient).max() > 0
assert np.array_equal(taught.path.gains, gains)
```

A gain that sets weights freely amplifies every sense against the belief; a weighing that
sums to the number of blocks (`blocks * softmax` of a steering patch's outputs, for
instance) moves weight from one sense to another. The map from a steering patch's outputs
to the gains, and its Jacobian for the seam below, are the application's.

## The readback of a moment

`BeliefPath` carries, per moment, what a governor or a steering patch reads and what a
page draws: `evidence` `(batch, time, encoded)`, the encoded evidence the repair and the
store read, after the gains; `code` `(batch, time, cells)`, the store's plain code at the
final reading; `gains` `(batch, time, blocks)`; `residual_alone` `(batch, time, blocks)`,
the repair map's move from the expectation with only that block heard and the store read at
zero, one evaluation of the map per block, computed with `probe=True`; and `surprise`
`(batch, time, blocks)`, each block's reading against the reading the previous belief's
slow readout implies, a root mean square over the block's compared channels in the block's
persistence units (a surprise of one means the belief predicted the sense no better than a
belief that expects the reading to stay as it was). `readback` also carries `evidence`
`(batch, encoded)`, the encoded evidence before any gain, `tanh(port(o) + e_b)`, and
`encode(observations)` gives it for any `(..., inputs)` reading: a surprise says that a
sense disagrees, the encoded evidence says what the sense is reading, and a steering patch
that decides a block's gain reads it before the gain it sets — learning to
capture one sense took reading that sense's evidence beside the surprises. The
surprise needs a declaration,
`set_implied_reading(implied, units)`: the map from the outputs to the reading each block
should give, with the channels it does not imply left `NaN`, and the mean squared change of
the compared channels from one moment to the next on a batch of training data. A row that
reads nothing has a zero probe and a zero surprise. `readback(observations, actions,
state=)` gives one moment's expectation, probe and surprise before its repair: a steering
patch reads it, sets the moment's gains, and the moment is then assimilated under them.

```python
def implied(outputs):
    reading = np.full((len(outputs), 8), np.nan)   # channels the map leaves NaN are not compared
    reading[:, :2] = outputs                       # the eye's first two channels read the predicted position
    return reading

cortex.set_implied_reading(implied, units=[0.01, 0.01])   # the persistence error of each block's compared channels
cortex.reset()
path = cortex.assimilate(o, a, gains=gains, probe=True)
assert path.residual_alone.shape == (4, 6, 2) and path.surprise.shape == (4, 6, 2)
assert path.evidence.shape == (4, 6, 12) and path.code.shape == (4, 6, 128)
assert np.all(path.surprise[:, :, 1] == 0.0)       # the ear has no implied channels
moment = cortex.readback(o[:, 0], a[:, 0], state=np.zeros((4, 8)))   # before the first moment's repair
assert np.allclose(moment.residual_alone, path.residual_alone[:, 0])
assert np.allclose(moment.surprise, path.surprise[:, 0])
assert np.allclose(moment.evidence, cortex.encode(o[:, 0]))              # what each block reads, before the gains
```

The surprise of a stream's first moment compares the reading against the readout of the
boundary belief, a zero belief when no `state` is given; a life that carries its boundary
compares against what its last belief implied.

The probes cost one evaluation of the repair map per block and moment; the
surprise costs one call of the declared map. A steering patch reading the
surprises alone has matched one reading the probes as well, so carry the
surprise first and ask for the probes when the task shows they help.

## An admitted step

`observe(rate=r)` moves the parameters by `r` times the adjoint. On a loss that is flat
around the mean predictor and steep once the readout binds, every fixed rate that learned
diverged within a few dozen chunks and the rates that stayed finite learned nothing. With
a target the step is admitted: the chunk is replayed from the same boundary under the
proposed parameters, with the store as it stands, and the largest halving of the start
whose replay lowers the chunk's loss by the Armijo margin is taken, sixteen halvings at
most; the store is written after the admission. The start is twice the last admitted step,
at most `rate`, so `rate` is a ceiling and the patch finds its own step within it;
`step_size` holds the last admitted step, travels with the snapshot, survives `reset()`
and is dropped by `reset_step()`. `accepted_rate`, `final_loss` and `replay_calls` are on
the observation. The admission keeps no moments of the gradient. It is the default because on
every measured lane the plain step either diverged or learned nothing, and
because restarting from twice the last accepted step costs one replay and finds
the rate the chunk can take.
`backtrack=False` takes the plain step at `rate`; a step on an external gradient
(`output_gradient=`) is plain, since the library can replay only the loss it can see.

```python
cortex.reset()
admitted = cortex.observe(o, a, y, rate=100.0, write=False)
assert admitted.updated and admitted.final_loss < admitted.initial_loss
assert 0 < admitted.accepted_rate < 100.0 and admitted.replay_calls >= 1
assert cortex.step_size == admitted.accepted_rate          # the next chunk starts from twice this
```

A rate far above what the chunk can take diverges within two chunks under the
plain step and is admitted at a small fraction of it, with several replays on
the first chunk and fewer on the next, since each start follows the last
admitted step. The loss never rises on an accepted step, and a step accepted at
the first replay costs one forward pass more than the plain step.

## A life lived online

`assimilate` and `observe` take `state=`, a boundary to start the moments from instead of
the live belief, and `keep_live=True` leaves the live state where it was. A life that has
stepped through a window one moment at a time, so a page could draw it, learns from that
window by replaying it from the belief that was live at its first moment, without losing
its place in the stream; `state` is the boundary it kept, `keep_live` keeps the present.

```python
cortex.reset()
cortex.assimilate(o[:, :3], a[:, :3])                    # the life steps through three moments
boundary, here = np.zeros((4, 8)), cortex.state          # the boundary it kept, and where it is
cortex.observe(o[:, :3], a[:, :3], y[:, :3], rate=1.0, write=False, state=boundary, keep_live=True)
assert np.array_equal(cortex.state, here)                # the step was taken, the place kept
```

## The cost of a moment

`macs_per_moment(probes=False)` counts the multiply-accumulates of one moment: the port,
the transition and its gate, the store's reads and the repair map per iteration, the slow
readout and the store's decode, and with `probes=True` one evaluation of the repair map per
block. `cost` counts what the patch has computed since `reset_cost()`: every moment
assimilated, observed, imagined or replayed in an admission, their multiply-accumulates,
and the admission's replays. Every arm of a comparison at matched compute reads the same
counter, and a composition adds its own readback work. A map port counts reuse of
its kernel at every output position. `readback_macs(probe=True)` estimates one
separate readback; a direct `readback()` call does not increment the patch's counters.

These counters estimate dense forward matrix work. They exclude the backward
adjoint, optimizer/admission bookkeeping, record writes, nonlinearities, sorting,
application callbacks and memory traffic. They cannot alone establish equal
training compute. `imagine` uses the same conservative moment model even though
it performs fewer operations. Report these conventions and measure total training
time and memory separately; see [contracts](contracts.md).

```python
cortex.reset_cost()
cortex.reset()
taught = cortex.observe(o, a, y, rate=100.0, write=False)
assert cortex.cost["moments"] == 4 * 6 * (1 + taught.replay_calls)
expected = 4 * 6 * (cortex.macs_per_moment() + taught.replay_calls * cortex.macs_per_moment(surprise=False))
assert cortex.cost["macs"] == expected  # admission replays omit surprise diagnostics
```

## A weight per moment and a mask per row

`observe(loss_weight=)` weighs each moment's error, `(time,)` or `(batch, time)`, and the
loss and the adjoint normalize by the weight's sum instead of the moment count, so a weight
of ones is the plain loss. A moment of weight zero is neither taught nor written: a jump of
tens of units on the predicted channels (a dot that appears or vanishes) would otherwise
carry more gradient than the chase around it, and its residual would be written into the
cells a quiet field activates, so that the filled store predicted appearances everywhere.
`observed=` is accepted per row as `(batch, time)` as well as per moment as `(time,)`, in
`assimilate`, `observe` and `imagine`; a row that observes nothing keeps its expectation
while the other rows repair, so streams whose screens come down at different moments share
a batch.

```python
weight = np.ones((4, 6))
weight[:, 2] = 0.0                       # the jump moment: neither taught nor written
observed = np.ones((4, 6), dtype=bool)
observed[1, 3:] = False                  # one stream's screen comes down at moment 3
cortex.reset()
masked = cortex.observe(o, a, y, rate=0.0, write=True, loss_weight=weight, observed=observed)
assert masked.writes == int(np.sum(observed & (weight > 0)))
assert np.allclose(masked.path.belief[1, 3], masked.path.expectation[1, 3])   # a row that reads nothing keeps its expectation
assert masked.path.residual[0, 3] > 0.0
```

## Teaching through a seam

A steering patch has no target of its own: its outputs are the gains of another patch's
moment, and what it should have done is what lowers that patch's loss. `observe(...,
output_gradient=dy)` in place of `target` takes an external gradient on the outputs `(batch,
time, outputs)` and carries it through the same backward scan into the parameters and the
gains; nothing is written and no loss is reported. The cortex's `gain_gradient`, passed
through the Jacobian of the caller's map from the steering patch's outputs to the gains, is
that `dy`. With the gradient of the target form, the parameter and gain gradients equal the
target form's exactly.

```python
steering = BeliefPatch(StructuredPort(4, [DenseBlock(0, 4, 6)]), actions=1, belief=6, outputs=2, cells=64, active=4, record_width=4, seed=2)
readbacks = np.concatenate([path.residual_alone, path.surprise], axis=-1)   # (N, T, 4): what the steering patch read
steering.reset()
weighing = steering.assimilate(readbacks, a)                                # its outputs set the gains through the caller's map
d_outputs = taught.gain_gradient                                            # the cortex's loss into the gains, times the map's Jacobian (one for the identity)
steering.reset()
seam = steering.observe(readbacks, a, output_gradient=d_outputs, rate=0.5)
assert seam.updated and seam.writes == 0 and seam.path.loss is None
```

The admission of a joint step (the cortex and the steering patch moved together, one
replay of the cortex's chunk deciding) stays with the caller: `backtrack=True` needs a
target, since the library can replay only the loss it can see. A readback that depends on
the previous moment's prediction is taken as given in the steering patch's adjoint.

## What it does not do

It has no certificate: the repair map is nonlinear and its contraction is measured, not
proved. It does not learn its keys; the store's code is the fixed sparse code of
`Records`. It does not partition the store by action or provenance; those are the
application's declarations. Categorical outputs and the night (`dream`, `sleep`) are the
record patch's; the belief patch's readout is linear.
