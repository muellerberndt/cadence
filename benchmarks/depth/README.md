# Depth on a task that needs it: the AND-count

Every observation is 24 pairs of bits. A pair is on when both its bits are; the answer is the
number of pairs on, less three, capped at six: seven classes, a uniform-random choice right a
seventh of the time. A pair's AND is a threshold of two inputs and the answer bins a count of such
thresholds, so no linear readout of the bits names it: one hidden layer approximates the task and
two represent it. A tanh network of one hidden layer of 128 trained by backpropagation reaches
about 0.43 of the held-out rows in 4,000 lessons of 64, one of two layers (128, 64) about 0.88:
depth pays here, and quickly.

[`and_count.py`](and_count.py) teaches, on the same lessons:

| arm | what it is |
| --- | --- |
| `one` to `four` | `Brain.compose(48, 7, modules=...)` with (128,), (128, 64), (128, 64, 64) and (128, 64, 64, 64): the apical compartment (`apical=0.25`) at the teaching point `"deep"` (the normalized point with the projection-relative floor 0.1 and nudged phases of a whole 32 sweeps), every row taught, no memories |
| `mlp_one`, `mlp_two` | tanh networks of hidden layers (128,) and (128, 64), backpropagation with Adam at 1e-3 on the softmax cross-entropy |
| `random` | the uniform-random choice |

Each lesson draws 64 of the 198,000 training rows; every 500 lessons the 2,000 held-out rows are
answered, a Cadence brain's answer by its own free settle.

```sh
CADENCE_FUSED=0 python benchmarks/depth/and_count.py --out /tmp/depth --lessons 4000 --seeds 0 1 2 --jobs 15
python benchmarks/depth/and_count.py --verify /tmp/depth/summary.json
```

The receipt `summary.json` holds every reading, the settle work (mean free sweeps and the share of
lessons that ran out of their free budget) and the depth gains over one region, bound to this
script's and the library's source; `--verify` recomputes the summary from the readings.

## Result

The historical [receipt](results/and-count.json), three seeds per arm, final held-out accuracy
after 4,000 lessons:

| Arm | Seeds 0, 1, 2 | Mean |
| --- | --- | --- |
| `one` | 0.63, 0.56, 0.57 | 0.58 |
| `two` | 0.64, 0.64, 0.70 | 0.66 |
| `three` | 0.85, 0.80, 0.77 | 0.81 |
| `four` | 0.88, 0.88, 0.84 | 0.86 |
| `mlp_one` | 0.43, 0.43, 0.42 | 0.43 |
| `mlp_two` | 0.88, 0.89, 0.89 | 0.89 |
| `random` | 0.14, 0.15, 0.15 | 0.15 |

Every added module raises the mean accuracy, and four modules finish about 2.2 percentage points
below the two-layer network. These accuracy readings do not establish comparable training speed.
One region already beats one hidden layer: its loop through the readout pair settles
the answer and the region together, a recurrent depth a feed-forward layer lacks. A free settle
takes 94 to 141 sweeps and none ran out of its budget. The receipt was produced on the NumPy
loop (`CADENCE_FUSED=0`) on a 192-core Linux machine with all 21 cells in parallel. Other
machines, and the default fused kernel on that machine, reproduce the pattern, not every digit,
since the lessons' arithmetic differs in its last bits: the fused kernel there reads one 0.579,
two 0.642, three 0.804 and four 0.853.

## Why the compartment

Without it the composed brain's reciprocal pairs sum their top-down half like any synapse. The loop
through the readout pair then grows with the weights until the settle slows near its critical gain,
the region below the motor region carries an echo of the answer, and a second module adds little.
The [apical compartment](../../docs/anatomy.md#the-apical-compartment) bounds every region's
top-down drive to a share of its bottom-up drive. Credit reaching a region k levels below the
output then shrinks about fourfold per level: the projection-relative floor lets each projection
step on its own scale, and nudged phases of a whole budget carry the nudge down, while the opposite
phases' common drift cancels in the contrast.

## Memory on deep brains

[`memory.py`](memory.py) runs the [look-back, contextual-choice and held-heads
benchmarks](../attention/README.md) unedited, with every brain they compose deepened: each
module width repeated two or three times, the apical compartment (`apical=0.25`) and the
teaching point `"deep"` (the normalized point with the projection-relative floor 0.1 and nudged
phases of a whole 32 sweeps). The retained receipt was taken before the point, with the floor
alone (`learning_normalize_relative=0.1`) beside normalized teaching where a benchmark set none.
Each benchmark's own gates and controls decide each founder.

```sh
CADENCE_FUSED=0 python benchmarks/depth/memory.py --out /tmp/depth-memory --depths 2 3 --jobs 32
python benchmarks/depth/memory.py --verify /tmp/depth-memory/summary.json
```

Founders passing their benchmark's gates ([receipt](results/memory.json)):

| Benchmark | Two modules | Three modules |
| --- | --- | --- |
| look-back, `first` and `last` | 4 of 4 | 4 of 4 |
| contextual choice | 3 of 4 | 4 of 4 |
| held heads, two facts reported | 4 of 4 | 4 of 4 |
| held heads, one action from two facts | 3 of 4 | 2 of 4 |

Every founder answers both look-back cues at 1.00 at both depths, and every held-heads founder
reports two facts. The misses: two-module contextual-choice founder 1 reads the right record
0.82 against 0.18, and its readout answers one class for both contexts; the one-action held-heads
rule misses where the two heads read the same record (founder 0 at both depths, founder 3 at
three modules). Founder 0 at two modules separates its heads with twice the lessons, at 0.885 on
untaught pairs against the gate's 0.90: a deeper brain acquires the rule more slowly.

These retained results use their original sources and original head gates, which measured
maxima among role records only. Current held-head runs measure maxima over the whole bank and
whether they identify the two asked facts. The original scores cannot decide that stronger
measurement retrospectively. Current workers also restore their temporary depth constructor
after every job, so a reused worker runs each cell at its declared depth.

Verify historical custody and recorded arithmetic explicitly from the repository root:

```sh
python benchmarks/provenance.py --historical benchmarks/depth/results/and-count.json
python benchmarks/provenance.py --historical benchmarks/depth/results/memory.json
```

This checks the registered result and archived checker; it does not retrain a founder or admit
the result as a measurement of the current implementation. New receipts bind every library
module, protocol, configuration, worker identity and declared cell census.
