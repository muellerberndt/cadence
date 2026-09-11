# Backends, devices, precision

```python
import cadence as cd
cd.available_backends()
# {'cpu': 'numpy float64', 'torch': 'mps float32', 'mlx': 'gpu float32'}   # an M-series Mac with both installed
```

| backend | where it runs | precision | install | use it for |
|---|---|---|---|---|
| `"cpu"` | NumPy, and the fused numba kernel | float64 | `cadence-net[fast]` | receipts, conformance, anything you will cite; every readout |
| `"torch"` on CUDA | NVIDIA GPU | float64, or float32 with `precision="float32"` | `cadence-net[accel]` | training at scale; receipts too, in float64 |
| `"torch"` on MPS | Apple silicon GPU through Metal | float32 | `cadence-net[accel]` | training on a Mac when torch is what you have |
| `"mlx"` | Apple silicon GPU through MLX, unified memory | float32 | `cadence-net[apple]` | training on a Mac: the faster of the two Apple paths |
| `"torch"` on CPU | torch CPU | float64 | `cadence-net[accel]` | one code path on a box without a GPU |

Every backend does the same arithmetic: the block transport of the wiring (dense blocks
between the owner ranges the named sets cut; a range that did not move keeps its product),
then one owner-local update, a nudge, adaptation, a mask, a tolerance, and `repair`. The
device backends keep the settled state on the device (`state.device`) so that a phase that
continues from it starts there, and the learning rule's contrast is read on the device
(`Settlement.contrast_on_device`): only one number per overlap comes back to the host.
Large wirings whose blocks do not fit `dense_limit` settle by the segmented scatter on
`"cpu"` and `"torch"`; `"mlx"` needs the blocks.

## Which hardware

**Apple silicon.** Two paths. `"mlx"` runs the block products and the owner updates as MLX
graphs on the unified memory and is the faster one: on an M4, one settlement step of a
language-model net (2,771 owners, 512 rows) takes 15 ms on MLX against 22 ms on torch/MPS,
and a whole learning update (three settlements and the contrast) 0.7 s against 1.1 s.
Both are float32; MPS has no float64 and MLX's GPU has none either, so a receipt readout
runs on `"cpu"`. The fused CPU kernel on the same net does one step in about 20 ms on one
performance core when the machine is quiet (the numbers above were taken on a machine at
load 20, which slows the CPU path more than the GPU paths). Rule of thumb: below about a
thousand moving owners the CPU kernel wins on latency; above it, MLX.

**Intel and AMD CPUs.** The `"cpu"` backend: NumPy float64 with the fused numba kernel
(`cadence-net[fast]`, which brings numba and scipy's BLAS). The kernel's cost per step is
one exponential per moving owner plus the block products; on x86 numba can vectorise the
exponential through SVML when Intel's `icc_rt` is installed (`pip install icc_rt`), which
roughly doubles the elementwise part. Set `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS` to the
cores you mean to use; the receipts record them (`cadence.timing.environment`).

**NVIDIA.** `"torch"` with CUDA. Float64 is the default and is what a receipt wants; on a
data-centre part (A100, H100) float64 runs at full rate. On a consumer or inference part
(L4, L40S, A10G, RTX) float64 is a thirty-second of the float32 rate, so train with
`Settlement(..., precision="float32")` and read out on `"cpu"`; the trainer in the author
repository does exactly that and records the conformance between the two. One host sync
per step (the tolerance) costs about 20 µs on CUDA and is negligible above a few hundred
owners.

**Several GPUs.** Not yet: one net settles on one device. The addition is a summed contrast
across devices, which is small; it is on the roadmap for the level-3 author.

## Choosing a device

```python
cd.Settlement(w, rule, backend="torch")                          # cuda, else mps, else cpu
cd.Settlement(w, rule, backend="torch", device="cpu")             # force
cd.Settlement(w, rule, backend="torch", precision="float32")      # speed on a consumer GPU
cd.Settlement(w, rule, backend="mlx")                             # Apple silicon through MLX
```

A learner built on a device engine trains there; `Learner.load(path, backend="cpu")`
brings a checkpoint back to the receipt backend, whatever trained it.

## Precision matters

Owners can sit on knife edges, where a difference of 1e-7 in a drive flips a bistable
readout. Float32 summation order alone did that to a motor neuron in the fly brain. Two
habits keep this honest:

1. Make receipts on `"cpu"` or on CUDA float64.
2. When you use MPS float32 for a page or a demo, run `cd.conformance` on the same wiring
   and clamp, and show the deviation. It is usually around 1e-5; when it is not, a readout
   near threshold is telling you something.

## Extending to another device

The two device kernels, `settle._TorchKernel` and `settle._MlxKernel`, are each one class
with the same five operations: block products between owner ranges (a matrix product per
pair), the elementwise owner update, a softmax over the nudged group, an equality check on
the still ranges, and a max over the movement for the tolerance. Any array library with
those five can host a backend; the owner-by-owner reference and `conformance` are what
you check it against, and `tests/test_settle.py` has the test each kernel passes.


## The fused kernel

With numba installed (`pip install "cadence-net[fast]"`) the CPU backend settles blocked
wirings in one compiled loop: the block transport, then every owner's repair, activation,
adaptation and nudge in place, the same float64 arithmetic in the same order as the NumPy
loop (checked to 2e-16). Two more things it does are exact for the same reason: an owner
whose potential did not move keeps the activation it published, and a range that hears
nothing is skipped for good once it is still, because such an owner's update is a fixed
function of its own state. It is used automatically when the wiring is blocked and no
trajectory is requested; `CADENCE_FUSED=0` in the environment forces the NumPy loop. The
owner-by-owner reference and `conformance` are unchanged and remain what any kernel is
measured against.

## Timing a decision

`cadence.timing.latency(decide)` calls `decide()` a thousand times and reports the median,
the 90th and 99th percentiles and the maximum in microseconds, with the number of voluntary
and involuntary context switches the scheduler made during the measurement (from
`getrusage`), and `environment()` records the thread limits, the cores the process is
pinned to when the platform can say (Linux), the load average and the library versions. A
wall-clock number in a receipt is a fact about a program on a machine on a day; this is the
machine's half of it. A tail far above the median with many involuntary switches is the
machine, not the net.
