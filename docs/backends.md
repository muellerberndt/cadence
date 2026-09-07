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

Small wirings are the other regime: below `dense_limit` owners (2048 by default) the NumPy
backend does the same sum as one product against the dense overlap matrix, because at that
size the interpreter overhead of the scatter would dominate. A learned net of a few hundred
owners settles in tens of microseconds per step that way, and the torch backend does the
same with one matrix product on the device. The result is identical to rounding;
`Settlement(..., dense_limit=0)` forces the segmented path on either backend.

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
