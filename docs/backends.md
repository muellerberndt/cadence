# Backends, devices, precision

```python
import cadence as cd
cd.available_backends()
# {'cpu': 'numpy float64', 'torch': 'mps float32'}   # on an M-series Mac with torch installed
```

| backend | where it runs | precision | use it for |
|---|---|---|---|
| `"cpu"` | NumPy | float64 | receipts, conformance, anything you will cite |
| `"torch"` on CUDA | GPU | float64 | large wirings at receipt precision |
| `"torch"` on MPS | Apple silicon GPU | float32 | interactive work; MPS has no float64 |
| `"torch"` on CPU | torch CPU | float64 | when torch is installed and you want one code path |

Both backends do the same arithmetic: one scatter of every overlap's message into its
owner's inbox per step, then one owner-local update. NumPy uses a segmented sum; torch uses
`index_add_`. Neither needs a dense matrix, so a wiring of a few million overlaps settles
in tens of milliseconds per step on a GPU and under a second on a CPU.

Small and layered wirings are the other regime. When the dense blocks of the wiring fit
(at most `dense_limit` squared entries, 2048 squared by default) the NumPy backend does the
same sum as block matrix products, and the torch backend does the same on the device. The
result is identical to rounding; `Settlement(..., dense_limit=0)` forces the segmented path
on either backend.

## The block transport

A layered net is mostly empty space. Its input owners hear nothing, its hidden owners hear
the inputs and the outputs, and the inputs, once clamped, stop moving after the first step.
One dense `n x n` product per step paid for all of that space and re-multiplied the still
inputs every step: on MNIST the cost of a step barely depended on the hidden width, because
the 784 input owners dominated it.

`cadence.blocks` cuts the owners into contiguous ranges at the boundaries of the wiring's
named sets (a set that is not one contiguous run is ignored) and keeps one dense block per
ordered pair of ranges that carries at least one overlap; `layered` and `embedded` give the
ranges directly. Each step, the inbox is the sum of the block products, and the product of a
range whose activation is bit-for-bit what it was at the previous step is reused. Nothing an
owner reads changes: the inbox is the same sum of the same messages in a different
association order, and `conformance` against the owner-by-owner reference still holds to
rounding. `engine.layout` describes the cut (`to_dict()` reports ranges, blocks and
entries); a wiring with no contiguous sets gets one block, the full matrix, as before.

On the MNIST shape (784 inputs, 256 hidden, 10 outputs, batch 256) one learning update went
from 63 ms to 18 ms on one M4 core, and at 32 hidden owners from 39 ms to 7 ms; the cost of
a settlement step now scales with the owners that move.

## Choosing a device

```python
cd.Settlement(w, rule, backend="torch")               # cuda, else mps, else cpu
cd.Settlement(w, rule, backend="torch", device="cpu")  # force
```

## Precision matters

Owners can sit on knife edges, where a difference of 1e-7 in a drive flips a bistable
readout. Float32 summation order alone did that to a motor neuron in the fly brain. Two
habits keep this honest:

1. Make receipts on `"cpu"` or on CUDA float64.
2. When you use MPS float32 for a page or a demo, run `cd.conformance` on the same wiring
   and clamp, and show the deviation. It is usually around 1e-5; when it is not, a readout
   near threshold is telling you something.

## Extending to another device

The torch kernel is one small class, `settle._TorchKernel`, with three operations: gather
`s[pre]`, scatter-add into `inbox`, and the elementwise update. Any array library that
offers those three can host a backend; the reference engine and `conformance` are what you
check it against.


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
