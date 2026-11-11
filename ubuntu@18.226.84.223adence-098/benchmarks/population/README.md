# Population throughput

A frozen reading. The population kernel, `cadence.population.PopulationPatch`, and this
instrument's script, `throughput.py`, left the library with the record patch; release 0.84.3
has both, and a checkout of `v0.84.3` reruns the measurement with the commands below. This
folder keeps the receipt of 2026-09-25.

`throughput.py` measures the population kernel against the NumPy patch on one processor core,
and writes `receipt.json` beside itself.

The work is one moment per brain and stream: an imagined reading, a one-moment path from
rest, and an observation with its write, on a patch of 32 inputs, 32 channels, 8 outputs and
a store of 256 cells (8 active), in a fixed random linear world per stream. The population
runs instances x streams in lockstep on the torch device (cuda, then mps, then cpu); the
reference runs the same patch, one brain in one stream with its own store, on one core with
the BLAS thread count at one. A second reference row runs 64 streams per call through one
shared store, which is a different world and is reported only for scale. The receipt also
holds the parity of the two paths on the cpu in float64: the reading and the slow step after
one observed moment.

```bash
python benchmarks/population/throughput.py              # the first available device
python benchmarks/population/throughput.py --device cpu --steps 20
```

The receipt of 2026-09-25 on an Apple M4 laptop (torch 2.14, the graphics processor):

| instances x streams | moments per step | moments per second |
| --- | --- | --- |
| 8 x 64 | 512 | 127,840 |
| 32 x 64 | 2,048 | 368,839 |
| 64 x 128 | 8,192 | 498,436 |
| 64 x 256 | 16,384 | 423,054 |

The reference, one brain in one stream on one core, ran 1,276 moments per second at the
same work (18,783 with 64 streams through one shared store); the best population row is
390 times the reference. This is a device/implementation throughput ratio at one patch
shape, not a matched-task advantage over another model family. CPU float64 parity errors
are below 2.23e-16 for the reading and 2.78e-17 for the slow step; these checks do not bound
the error of the timed GPU path. The 16,384-stream configuration has 139 MB of store state,
but the timing alone does not identify the cause of its lower throughput. Each row times
40 steps with no repeated-run uncertainty estimate. Wall-clock numbers depend on the
machine; the receipt records the substantial background load at the end of this run.
