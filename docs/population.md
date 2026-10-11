# Many brains at once: populations on a device

<!-- requires: torch -->

One class runs a whole population of record patches on a graphics processor (or any torch
device): `PopulationPatch`. Every brain is an instance with its own parameters; every
world a brain is in is a stream with its own record store; every settle of the population
is one batched product.

```python
import torch
from cadence import RecordPatchNet
from cadence.population import PopulationPatch

net = RecordPatchNet(20, 16, 3, groups=(3,), cells=512, active=8, slowest=2.0)
pop = PopulationPatch.from_patch(net, instances=8, streams=32)   # 8 brains, each in 32 worlds
x = torch.rand(8, 32, 20, device=pop.dev)                          # one reading per brain and world
out = pop.imagine(x)                                               # out["output"]: (8, 32, 3)
target = torch.nn.functional.one_hot(torch.randint(0, 3, (8, 32)), 3).to(pop.dev, out["output"].dtype)
pop.observe(x, target, rate=0.3)                                   # every brain its own step, every world its own write
pop.inherit(torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]), sigma=0.1)      # selection: copies of the parents, mutated
```

The device is chosen for you (cuda, then mps, then cpu). `from_patch` starts every brain as
a copy of a NumPy patch, which is how a schooled patch becomes a population; the plain
constructor takes the same arguments as `RecordPatchNet` plus `instances` and `streams`.

## What batches, and what does not

Brains, worlds and (through several `PopulationPatch` objects) the patches of a brain all
batch, because nothing crosses them: each brain's slow step is the adjoint of its own
one-moment loss, each world's records take their own residual, and no gradient reaches a
store or another patch. The moments of one world do not batch; a temporal patch settles
them in order, as any recurrent system does. This class is the twin of a one-moment path
from rest, the reading a decision or a school batch gives a patch.

## Parity and measurement

The tests hold the twin to the NumPy patch on the cpu in float64: the prediction, the slow
step against the NumPy adjoint (linear and categorical) and a write with its read, all to
1e-9. `benchmarks/population/throughput.py` measures the kernel on one moment's work per
brain and stream (an imagined reading and an observation with its write) and writes a
receipt beside itself. On an Apple M4 laptop's graphics processor a patch of 32 channels
with a store of 256 cells ran 8,192 streams at 498,000 moments per second and 16,384 at
423,000, against 1,276 for the same patch on one processor core (390 times). That ratio
compares one device and implementation with a sequential reference at one patch shape; it
is not an advantage over another model family, and it does not transfer to other shapes or
to a whole training run. The two paths agree on the reading to 2e-16 in float64 on the cpu,
which does not bound the error of the timed graphics-processor path. Each row times 40
steps with no repeated-run estimate. Batching brains and streams into one tensor program
needs their parameters to be independent along the population axis; a shared global loss
across examples of one model batches as well, so batching is a layout, not a property of
local learning. The tables cost `instances * streams * cells * outputs` values; keep stores
small and brains many.

`observe` stages every change and commits none on a bad moment: the inputs must be finite
and the targets finite on the unmasked moments (a masked-out moment carries no lesson, so its
target is ignored and may be NaN); the moved statistics, the rows the write would leave and
the parameters and Adam moments the slow step would leave are checked before any of them
lands, and a moment that would leave a nonfinite table, statistic, parameter or moment
raises `ValueError` with nothing changed. A write whose rows could overflow is summed on a
copy of those rows first. The common case pays one pass over the rows the write read, one
over the residual, a few small kernels and one device sync: on an M4, one observe of 64
brains in 2,048 streams (a patch of 64 inputs, 64 channels, 16 outputs, 256 cells) took
147 ms against 134 before the checks on the graphics processor and 338 against 318 on one
core; at 8 brains in 1,024 streams the graphics processor pays 23 percent, the launch cost
of the small kernels.

## The kernel at the width of a game

A population that plays a game reads wide (a hold'em seat reads 283 ports; with the
swap, both players' actions and rewards, the readbacks of three slots and their settled
contexts the canonical reading is 1,414 ports at 256 hidden units) and lives through
episodes of unequal length. Six additions make that regime cheap; each keeps the parity
tests.

- The store is read and written through the active cells alone: a flat row index into the
  table viewed as rows, `index_select` for the read, `index_add_` for the write. A settle
  no longer touches every cell of every stream, so the store's size sets memory, not time.
- The adjoint of the one-moment loss is written out (six matmuls) and the dense path uses
  `matmul`; on MPS autograd through `einsum` cost four times as much.
- `imagine(inputs, stream_of=...)` and `observe(..., stream_of=...)` take several moments
  per stream in one call, `stream_of` naming each moment's stream as `(moments,)` or
  `(instances, moments)`. The drive's five imagined actions settle in one launch, and every
  queued lesson of every stream is taught in one observe (the writes of one stream land
  against the store as it stood, the NumPy patch's `write_batch`).
- `observe(..., mask=...)` gates the streams that have a moment to learn from: the slow
  loss of an instance is the mean over its masked moments, only those write and move their
  statistics, and an instance with no masked moment does not move. Hindsight learning over
  episodes of unequal length runs on it.
- `observe(..., weight=...)` reweights the squared errors of a linear readout per output.
  A patch that predicts a wide reading of which a few ports matter counts them by their
  weight. Normalizing per block, so every block a slot reads counts once, gives two outcome
  ports the weight of hundreds of sense ports; without it the prediction a drive chooses on
  receives a vanishing share of the repair.
- `record_weight` (instances, outputs) damps the records of an output whose target is a
  sample rather than a fact (a payoff), and `imagine` returns `familiarity`, how much of a
  reading's code falls on cells the stream's store has written before: whether this brain
  has been near this reading in this world, which a drive can read as the trust to place
  in an imagination.
- `optimizer = "adam"` scales the slow step by the adjoint's running moments per instance
  and parameter, the rate then being the learning rate. In the wide regime the context is
  small and a fixed-rate step barely lifts the weights out of their initialization over
  hundreds of moments; Adam on the adjoint moves them.

Timings on an Apple M4, 4,096 streams, an 830-port reading, 64 hidden units, 128 cells,
8 active: an imagined reading 74 ms before these changes and 15 ms after, an observation
with its write 213 ms and 41 ms.

## Training across several devices

Instances of a population are independent: nothing crosses them in a settle, a write or a
slow step. So a population splits across devices without communication during play, and
the recommended structure for a run larger than one device is one population per device
with a file as the only exchange. This is the structure; it has been run on one device,
not yet across several.

1. **School one founder on one device.** One brain in as many streams as the device holds
   (512 per instance on an M4): its slow step integrates over every stream, so the
   streams are the data rate, and tens of millions of observed moments fit in a couple of
   hours. Save its parameters and genes as the checkpoint every device starts from.
2. **One population per device, from the checkpoint.** Each device holds its own
   `PopulationPatch` objects (one per cortex slot) with its share of the brains, all
   starting as mutated copies of the founder, each brain in its own worlds. The world's
   tables (the world as arrays, its draws as a bank) are small and copied to every device.
3. **The hall as the exchange.** After every generation each device writes its best
   brains (parameters and genes) to a shared directory; the elders of the next generation
   on every device are drawn from the union of the halls. Brains never move during a
   generation; a few megabytes move between generations.
4. **Common draws across devices.** Seed the random sequences of the shared worlds by stream
   index, the same on every device, so the fitness of brains on different devices is a
   paired comparison.
5. **Size by the stores.** Memory is `instances * streams * cells * outputs` floats per
   slot, plus the lesson queues (`instances * streams * queue * outputs`, twice). At 1,414
   outputs and 128 cells, 64 brains in 32 worlds cost 4.4 GB of stores; 256 brains in 64
   worlds 35 GB; 1,000 brains in 64 worlds 139 GB. A store of 32 cells brings the
   thousand to 43 GB on one 80 GB device; otherwise the population splits four ways.
6. **One job per device.** Two populations on one device, or a memory cap that makes the
   allocator page, turn minutes into hours (measured on the M4: a generation of four
   minutes became forty).

What speeds up as brains converge is the adjustment, not the settle: a settle is one
fixed pass. To make compute follow convergence, gate the lessons by surprise (the brain
predicted its next reading at every decision; drop the lessons whose prediction was near
the truth before compacting the queue), so a converged brain against a familiar world
learns for almost nothing and a novel world costs the full step.
