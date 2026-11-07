# Lessons on a device

[`lessons.py`](lessons.py) times `Brain.lesson` (a free settle to the learner's tolerance, two
nudged phases, their contrast and one update) on two brains, at the batches you name:

| brain | what it is |
| --- | --- |
| `deep` | a ventral-stream anatomy over a 32 x 32 RGB retina: V1, V2 and V4 of unshared receptive fields, IT, an association region of 64 and five word slots of four; the top-down half of every reciprocal pair on the apical compartment (`apical=0.25`), normalized teaching, nudged phases of a whole 32 sweeps; 7,956 neurons, 903,520 synapses |
| `flat` | `Brain.compose(3072, 20, slots=(4,) * 5)` with every founder setting; 3,220 neurons, 203,264 synapses |

```sh
python benchmarks/torch_gpu/lessons.py --device cuda:0 --batches 64 256 1024 4096
python benchmarks/torch_gpu/lessons.py --device cuda:0 --precision float32
python benchmarks/torch_gpu/lessons.py --device cuda:0 --split --batches 64   # time per part
OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 python benchmarks/torch_gpu/lessons.py --backend cpu
```

Each row printed is one brain at one batch: seconds per lesson, the mean free sweeps and the
torch transport plan (its kind, groups, products, Triton programs and graph replays). The script
uses the public API only, so the same command times any library version.

## On an NVIDIA A10G

Seconds per lesson on one A10G that a training campaign shared throughout (other processes held
it at 50 to 100 percent utilization, so readings vary by up to a quarter from run to run; before
and after were run alternately, one process each, ten lessons at batches up to 256 and three
above). One CPU core is the `cpu` backend with Numba on the same machine. Before is 0.90.0.

| brain | batch | one CPU core | CUDA float64 before | CUDA float64 after | after, float32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| deep | 64 | 2.41 | 1.97 | 0.169 | 0.092 |
| deep | 256 | 8.71 | 2.56 | 0.536 | 0.225 |
| deep | 1,024 | 41.5 | 7.67 | 1.64 | 0.612 |
| deep | 4,096 | 171 | 25.7 | 6.22 | 2.33 |
| flat | 64 | 0.045 | 0.066 | 0.062 | 0.060 |
| flat | 256 | 0.172 | 0.158 | 0.107 | 0.098 |
| flat | 1,024 | 0.861 | 0.364 | 0.278 | 0.223 |
| flat | 4,096 | 3.66 | 1.36 | 1.00 | 0.775 |

The deep brain's float64 lesson on the device beats one CPU core by about ten times at batch 32
(1.54 s against 0.16 s) and by 14 to 28 times from batch 64 on. The flat founder never does: its
sweeps are mostly elementwise work over 3,072 sensory neurons, which the device does in float64
at memory speed and the compiled CPU loop does well; it gains at most four times.

The same lessons on an A10G that nothing else used, against 0.94.0 on that box (one process each, five
lessons after three to warm up), and one core of the same machine (an AMD EPYC 7R32 with Numba):

| brain | batch | one CPU core | CUDA float64 0.94.0 | CUDA float64 | CUDA float32 0.94.0 | CUDA float32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| deep | 64 | 2.27 | 0.073 | 0.055 | 0.033 | 0.027 |
| deep | 256 | 8.41 | 0.233 | 0.191 | 0.096 | 0.083 |
| deep | 1,024 | | 0.753 | 0.704 | 0.324 | 0.283 |
| deep | 4,096 | | 2.84 | 2.77 | 1.25 | 1.11 |
| flat | 64 | 0.024 | 0.014 | 0.013 | 0.012 | 0.013 |
| flat | 256 | 0.144 | 0.042 | 0.037 | 0.035 | 0.035 |
| flat | 1,024 | | 0.166 | 0.139 | 0.136 | 0.133 |
| flat | 4,096 | | 0.614 | 0.537 | 0.517 | 0.499 |

The first table above read 0.90.0 and its successor on a shared GPU.

Where a deep lesson's time goes (`--split`, the device synchronized around each part, seconds):

| batch | version | free settle | two nudged settles | residual checks | apical gain | contrast and update |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 64 | before | 0.194 | 1.46 | 0.124 | 0.115 | 0.054 |
| 64 | after | 0.057 | 0.101 | 0.011 | 0.006 | 0.021 |
| 4,096 | before | 3.66 | 16.3 | 1.03 | 4.89 | 0.382 |
| 4,096 | after | 1.06 | 3.73 | 0.429 | 0.079 | 0.499 |
