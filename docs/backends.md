# Backends, devices, precision

<!-- requires: torch -->

`TemporalPatchNet`, its planner and `TemporalMemory` currently use NumPy.
The optional backends below apply to the existing `NeuralGraph`/`PatchNet` graph
interfaces; installing an accelerator does not move temporal repair onto it.

Begin with the NumPy-only [installation](../README.md#get-started). Optional extras
add compiled CPU transport or accelerator libraries:

```bash
python -m pip install "cadence-net[fast]==0.81.0"
```

Replace `[fast]` with `[accel]` for PyTorch or `[apple]` for MLX. From a checkout,
use `python -m pip install -e ".[fast]"`. An installed accelerator library is
not enough to select it: pass `backend="torch"` or `backend="mlx"` to
`Brain.compose` or the lower-level `NeuralGraph`.

```python
import cadence as cd
cd.available_backends()
# {'cpu': 'numpy float64', 'torch': 'mps float32', 'mlx': 'gpu float32'}   # an M-series Mac with both installed
```

| backend | where it runs | precision | install | use it for |
|---|---|---|---|---|
| `"cpu"` | NumPy; optional Numba and SciPy | float64 | base install; `[fast]` for acceleration | first examples, numerical checks, sparse graphs |
| `"torch"` on CUDA | NVIDIA GPU | float64, or float32 with `precision="float32"` | `cadence-net[accel]` | large brains and long lives; receipts too, in float64 |
| `"torch"` on MPS | Apple silicon GPU through Metal | float32 | `cadence-net[accel]` | a Mac when torch is what you have |
| `"mlx"` | Apple silicon GPU through MLX, unified memory | float32 | `cadence-net[apple]` | an alternative Apple backend; benchmark the actual workload |
| `"torch"` on CPU | torch CPU | float64 | `cadence-net[accel]` | one code path on a box without a GPU |

The backends implement the same neuron equations with different kernels and precision.
For a blocked connectome, transport uses dense blocks between neuron ranges; unchanged
ranges can reuse their products. Each step then applies the neuron update, nudge,
adaptation, mask, and stopping check. The device backends keep the settled state on the
device (`state.device`) so that a phase that continues from it starts there, and the
learning rule's contrast is read on the device (`NeuralGraph.contrast_on_device`). With
resident phase states on blocked PyTorch graphs, `Learner` updates, contrast,
momentum and RMS normalization remain on the device. `ActorCritic` uses its
device update path when the required blocked
PyTorch phase states are resident and its own `momentum` and `normalize` are
both zero; its critic and dopamine calculations still use host arrays.
With either adaptive setting enabled, actor contrasts and traces
are read on the host and its optimizer runs in NumPy. The learner's adaptive
settings and the actor's adaptive settings are separate configurations.
Scalar step reports synchronize. Reading parameters or optimizer history,
saving a checkpoint, or entering a host-only path materializes the required arrays.
Public optimizer attributes remain mutable NumPy arrays; edits made through them are
picked up by the next update. History uses float32 on MPS and float64 on torch CPU/CUDA.
MLX contrast returns arrays to the host for the optimizer.

Torch and MLX factor the phase contrast as
`A_plus.T @ (B_plus - B_minus) + (A_plus - A_minus).T @ B_minus`.
This is the same bilinear difference as subtracting the two phase Gram matrices,
but avoids subtracting large products when the desired contrast is small. Torch uses
the corresponding product identity for each row's eligibility trace. The accelerator
path needs no host equality check. NumPy and Numba use the same factorization,
including when processing phase values restored from a checkpoint. Identical phases
give exactly zero contrast.
Input rounding, reduction order and cancellation between distinct contributions still
limit accuracy; float32 learning trajectories need not match float64 at a fixed
absolute error. RMS normalization can amplify small contrast errors. Gradient checks
should use float64 CPU/CUDA and declare the nudge size and phase residuals.

Large connectomes whose blocks do not fit `dense_limit` use sparse transport. The CPU backend
uses SciPy CSR when installed, and the NumPy segmented sum otherwise. PyTorch uses its
gather/scatter path; `"mlx"` needs the blocks. A records read is one dense product of the
sparse code with a table; `Records` runs on the host with NumPy whatever backend the brain
uses.

## Sparse CPU transport

A CSR row holds the synapses onto one neuron. The matrix multiplies the published
activations directly, avoiding the NumPy fallback's temporary array with one value per
batch row and synapse. This changes transport only: the neuron update, nudges, masks,
adaptation and stopping conditions are the same. Floating-point summation order can
change slightly, so compare complete trajectories at the precision a task requires.

SciPy comes with `cadence-net[fast]`; the NumPy-only installation remains supported.
Sparse indices are cached on the connectome, and each brain's matrix shares its current
weights. Sparse learner contrasts use bounded chunks on the host, including when their
settling phases ran through PyTorch gather/scatter; they do not allocate a dense Gram matrix. Replacing parameters drops the old matrix wrapper; the topology can be reused.
For a batch of size B with E synapses and N neurons, one avoided float64 message array
occupies `8 * B * E` bytes, while the result occupies `8 * B * N` bytes. The CSR index
cache adds approximately `4 * (E + N + 1)` bytes when 32-bit indices suffice.

`brain.to_dict()["transport"]` is `"dense"` for a blocked connectome and `"segmented"` for
a sparse one. `"sparse_kernel"` is `null` before a sparse CPU multiply runs, then
`"scipy_csr"` or `"numpy_segmented"`. Inspection does not instantiate a kernel.

## Which hardware

Choose using the actual changing-input workload. Small brains can spend more time
launching GPU operations than doing arithmetic; large batches and dense blocks can
benefit from accelerator matrix products. Sparse graphs require a separate measurement.
Warm up the backend and synchronize the GPU around wall-clock measurements.

The CPU backend uses NumPy float64, with optional numba and SciPy acceleration through
`cadence-net[fast]`. Cap BLAS threads when running independent experiment workers;
`cadence.timing.environment()` records thread settings. PyTorch supports CPU, CUDA and
Apple MPS, while MLX provides an additional Apple path. A single brain uses one
device. Neither backend choice nor parameter count establishes efficiency by itself.

CUDA defaults to float64; `precision="float32"` selects float32 settling. Hardware
throughput and the useful precision depend on the device and problem. Compare outputs,
residuals and learning curves with float64, especially near multiple equilibria. MPS
uses float32 for parameters and state because it does not support float64.

## Residual checks and synchronization

`NeuralGraph.equilibrate` chooses work from the measured potential/adaptation equation residual.
An unchanged warm state already within tolerance takes zero settling steps, but still costs
one residual check. `chunk` trades check frequency against overshoot: every row advances
together and the final check must meet the same tolerance regardless of chunk size. Step
count measures numerical settling work; it is not a measure of surprise or task quality.

For unread float64 Torch states, residual transport and reduction stay on the device and
only one scalar per row returns. Reading or editing the host potential/adaptation selects
the host reference, as does `residual(..., on_device=False)`. Float32 Torch and MPS states
also retain the float64 host reference, so faster checks do not quietly weaken its precision.
Accelerator block products recompute source products at each step without reading
`torch.equal` flags on the host. CPU blocks retain the exact source cache. The equations
stay the same. The activation uses one leaky-rectifier primitive for its two
linear branches. A positive movement tolerance still requires a scalar check every step;
fixed step counts and the inner chunks of `equilibrate` avoid that check.

To compare two implementations, run both on the same workload with identical dtype,
steps, tolerances and centered updates over several seeds; warm up, alternate their order,
synchronize the device around each timed call, and compare residuals, states and
parameters as well as time. A small fixed-topology timing says nothing about another
device, a comparison with an MLP, learning quality at scale or energy use in joules.

## Choosing a device

```python
connectome = cd.layered(4, 16, 2, seed=0)
neuron_model = cd.learning_neuron_model()
cd.NeuralGraph(connectome, neuron_model, backend="torch")                          # cuda, else mps, else cpu
cd.NeuralGraph(connectome, neuron_model, backend="torch", device="cpu")             # force
cd.NeuralGraph(connectome, neuron_model, backend="torch", precision="float32")      # speed on a consumer GPU
if "mlx" in cd.available_backends():
    cd.NeuralGraph(connectome, neuron_model, backend="mlx")                         # Apple silicon through MLX
```

Settling runs on the chosen device. Blocked PyTorch `Learner` updates use the
device optimizer; sparse contrasts, MLX optimization and the adaptive actor
path described above use host arrays. `Learner.load(path, backend="cpu")`
brings a checkpoint back to the receipt backend, whatever backend it learned on.

## Precision matters

CUDA block transport accumulates each matrix product directly into its destination,
preserving block order and the original qualification checks. The
[System 1 CUDA qualification](../benchmarks/system1_cuda/RESULTS.md) compares
source-pinned CPU, CUDA float64 and CUDA float32 execution on an RTX 4000 Ada
laptop, including independent derivatives, refusal, memory and checkpoint custody.
It reports complete decision costs and timing variation: the small continuing
nursery remains faster on CPU with Numba. These results do not qualify MPS or the
separate population solver, and do not establish speech or sequence throughput.

Near multiple attractors, a small rounding difference can change the selected
state. Compare the deployed backend against CPU float64 with `cd.conformance`
on representative drives, and report the measured deviation. Check residuals
and task readouts as well as trajectories. A receipt can record any backend;
it should declare precision and include the relevant numerical checks.

## Extending to another device

The two device kernels, `_TorchKernel` and `_MlxKernel` in `cadence.brain`, are each one
class with the same five operations: block products between neuron ranges (a matrix product
per pair), the elementwise neuron update, a softmax over the nudged group, an equality check
on the still ranges, and a max over the movement for the tolerance. Any array library with
those five can host a backend. Check it against the neuron-by-neuron reference and
`conformance`; `tests/test_brain.py` has the tests each kernel passes.


## The fused kernel

With Numba installed through `[fast]`, the CPU backend settles blocked
connectomes in one compiled loop: the block transport, then every neuron's potential
update, activation, adaptation and nudge in place. Floating-point implementations are
compared against the reference in the tests. Two cache rules avoid repeated work: a neuron
whose potential did not move keeps the activation it published, and a range that receives no
synapses is skipped for good once it is still, because such a neuron's update is a fixed
function of its own state. It is used automatically when the connectome is blocked and no
trajectory is requested; `CADENCE_FUSED=0` in the environment forces the NumPy loop. The
neuron-by-neuron reference and `conformance` remain what any kernel is measured against.

Settling accepts a shared mask of shape `(n,)` or `(1, n)`, or a separate
mask for each row with shape `(batch, n)`, across CPU and device paths. The
fused kernel reads a broadcast view, so sharing a mask does not allocate a copy
per row. Masks apply to potential and activation at each step; zero removes a
neuron's published activity, while any existing adaptation continues to decay.
Warm starts recompute the published activation under the current mask before
transporting it, including when a previous fractional mask is changed or removed.

## Timing a decision

`cadence.timing.latency(decide)` calls `decide()` a thousand times and reports the median,
the 90th and 99th percentiles and the maximum in microseconds, with the number of voluntary
and involuntary context switches the scheduler made during the measurement (from
`getrusage`), and `environment()` records the thread limits, the cores the process is
pinned to when the platform can say (Linux), the load average and the library versions. A
wall-clock number in a receipt is a fact about a program on a machine on a day; this is the
machine's half of it. Context switches can explain some timing variation;
profile before assigning a cause to a slow tail.

For a changing-input workload, `decide()` must advance an input sequence and carry or
reset state according to the deployment contract. Repeating an unchanged input measures
a stable-state fast path. Report cold and warm results, output quality, and residuals
separately; elapsed time on that fast path does not establish general settling speed.

On Windows, load averages and context-switch counts may be unavailable and are
reported as `None`. `latency` requires positive `repeats` and nonnegative `warmup`.
Synchronize asynchronous accelerator work inside the timed callable.

Use `brain.with_parameters(...)` for parameter changes. Assigning a complete `efficacy`
or `bias` array also refreshes the running backend. Do not mutate parameter or connectome
arrays in place: derived transport caches assume their values stay fixed. Rebuild a
connectome when changing topology. After a device update, always read `learner.brain`;
older NeuralGraph objects can share the updated kernel while retaining stale host copies.
