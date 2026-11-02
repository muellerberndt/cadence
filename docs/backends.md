# Backends, devices, precision

<!-- requires: torch -->

Every graph brain runs on the `"cpu"` backend in NumPy float64 unless you pick another. Start
from the NumPy-only [installation](../README.md#get-started) and add an extra:

```bash
python -m pip install "cadence-net[fast]==0.92.0"
```

`[fast]` installs Numba and SciPy, `[accel]` PyTorch and `[apple]` MLX; from a checkout, use
`python -m pip install -e ".[fast]"`. Installing a library never selects it: pass
`backend="torch"` or `backend="mlx"`. `available_backends()` lists the importable backends
with the device and default precision each would use.

```python
import cadence as cd

print(cd.available_backends())
# e.g. {'cpu': 'numpy float64', 'torch': 'mps float32', 'mlx': 'gpu float32'} on an M-series Mac
```

## Choose a backend

| backend | where it runs | precision | extra | use it for |
|---|---|---|---|---|
| `"cpu"` | NumPy, with Numba and SciPy when installed | float64 | none; `[fast]` to accelerate | first examples, receipts, numerical checks, sparse graphs, small continuing brains |
| `"torch"` on CUDA | NVIDIA GPU | float64, or float32 with `precision="float32"` | `[accel]` | deep brains, large batches, long lives |
| `"torch"` on MPS | Apple silicon GPU through Metal | float32 only | `[accel]` | a Mac when torch is what you have |
| `"mlx"` | Apple silicon GPU through MLX, unified memory | float32 | `[apple]` | an alternative Apple path for blocked connectomes |
| `"torch"` on CPU | torch on the host | float64, or float32 with `precision="float32"` | `[accel]` | one code path on a machine without a GPU |

`backend="torch"` picks CUDA, then MPS, then CPU; `device=` overrides it. One brain uses one
device. MPS refuses float64, `"mlx"` refuses a connectome whose blocks exceed `dense_limit`,
and a float32 backend refuses a neuron rule whose constants float32 cannot represent.

## What runs where

`backend=` (default `"cpu"`) and `device=` (default `None`, the best device) are construction
arguments of `NeuralGraph` and `Brain.compose`; `Brain.retune` refuses them. `precision=`
(`"float64"` or `"float32"`, read by torch, default by device) is an argument of
`NeuralGraph`, `Brain.load` and `Learner.load` only, so a composed brain runs at its device's
default precision until it is saved and loaded with another. Loads default to the saved
precision; `Brain.load` defaults to `backend="cpu"`, `Learner.load` to the saved backend.
`Learner.load(path, backend="cpu")` brings a checkpoint learned on any device back to the
float64 receipt backend.

The composed brain's memory stores are read on the host inside a device brain too.

On a device backend `settle_batch` returns a state that stays resident (`BrainState.device`;
`None` on the CPU and from the one-stimulus `settle`), so a continuing phase starts there.
The learning contrast is computed there too (`NeuralGraph.contrast_on_device`, `None` unless
both states are resident; MLX needs a blocked graph). Torch `Learner` updates, momentum and RMS
normalization stay on the device, and so does the apical gain a nudged phase holds
(`NeuralGraph.apical_gain`); MLX hands the contrast to a host optimizer. `ActorCritic` updates on the device when both of its
phases settled there; its critic and dopamine use host arrays. Reading
parameters or saving a checkpoint copies arrays back. Parameter edits and the `learner.brain`
rule are in [contracts](contracts.md).

## Choosing a device

```python
connectome = cd.layered(4, 16, 2, seed=0)
neuron_model = cd.learning_neuron_model()
cd.NeuralGraph(connectome, neuron_model, backend="torch")                        # cuda, else mps, else cpu
cd.NeuralGraph(connectome, neuron_model, backend="torch", device="cpu")          # force a device
cd.NeuralGraph(connectome, neuron_model, backend="torch", precision="float32")   # float32 settling
if "mlx" in cd.available_backends():
    cd.NeuralGraph(connectome, neuron_model, backend="mlx")
brain = cd.Brain.compose(8, 3, backend="torch", device="cpu")                    # a composed brain
check = cd.conformance(cd.NeuralGraph(connectome, neuron_model, backend="torch"), {0: 1.0})
print(check["backend"], check["max_abs_deviation"])                              # vs the float64 reference
```

## Which hardware

Use CPU with Numba for small continuing brains, and measure an accelerator on the actual
changing-input workload before switching: small brains spend more time launching GPU
operations than doing arithmetic, while deep brains and large batches gain from the device.
[`benchmarks/torch_gpu/lessons.py`](../benchmarks/torch_gpu/README.md) times `Brain.lesson` on
a deep apical visual brain (7,956 neurons, 903,520 synapses) and on the flat founder; on an
NVIDIA A10G shared with other work, CUDA float64 runs a lesson of the deep brain about 14
times faster than one CPU core at batch 64 and about 27 times faster at batch 4,096, while the
flat founder (3,220 neurons, 203,264 synapses) gains at most four times over one core: its
sweeps are elementwise work over the sensory neurons, which the device does in float64 at
memory speed. `precision="float32"` runs the deep brain's lessons two to three times faster
again (that GPU does float64 at a sixty-fourth of its float32 rate); check it against float64
as below before relying on it. The [System 1 CUDA qualification](../benchmarks/system1_cuda/RESULTS.md)
compares CPU, CUDA float64 and CUDA float32 on complete decisions; on its small continuing
nursery CPU with Numba is faster. Cap BLAS and Numba threads for parallel workers. To compare two
implementations, use identical dtype, steps and tolerances over several seeds, warm up,
alternate the order, synchronize around each timed call, and compare residuals and states too.

## Precision matters

Near multiple attractors a rounding difference can change the selected state, and float32
learning need not track float64 within a fixed error; RMS normalization amplifies small
contrast errors. Every backend forms the phase contrast as
`A_plus.T @ (B_plus - B_minus) + (A_plus - A_minus).T @ B_minus`, so identical phases give
exactly zero. Check the deployed backend with `conformance` on representative drives, compare
residuals and task readouts, and declare precision in any receipt. Run gradient checks on
float64 CPU or CUDA with the nudge size and phase residuals declared.

## Residual checks, masks and conformance

`NeuralGraph.equilibrate` chooses its work from the potential and adaptation equation
residual: a warm state within tolerance takes zero steps and one residual check. `chunk` trades
check frequency against overshoot at the same final tolerance. Step count measures numerical
settling work, never surprise or task quality. `residual(drive, state, on_device=True)` checks
an unread torch state on its device in float64, one scalar per row back (a float32 state
through a float64 copy of the kernel, the same float64 reference); MPS and MLX states, a state
read on the host and `on_device=False` use the float64 host reference. A positive movement
`tolerance` in `settle_batch` tests every step; on CUDA the test is read one sweep behind (the
next sweep is queued before the host reads the flag, into the other state slot), so the host
never stalls the device, and the reported state and step count are those of the first sweep
that met the tolerance. Fixed step counts and the inner chunks of `equilibrate` read nothing.
CPU and MPS check before starting another sweep. For a direct CUDA movement-tolerance solve,
`BrainState.steps` is the returned state's sweep index, `discarded_steps` counts lookahead,
and `attempted_steps` counts both. Use the latter for executed-work comparisons; this is
still a sweep count rather than a total resource or energy meter.

Torch workspaces reuse device temperature inputs when arousal changes the sampling heat.
Captured graphs read those current values. Topology and arithmetic-branch changes use bounded
caches whose eviction drops dependent configurations and captures together, so a continuing
changing-temperature life does not retain a new set of nudge tensors at every moment.

Masks have shape `(n,)` or `(1, n)` for every row, or `(batch, n)` per row, with entries in
[0, 1], on every backend. A mask scales potential and activation at each step; zero removes a
neuron's published activity while its adaptation decays. A warm start recomputes the published
activation under the current mask, and the fused kernel reads a shared mask without copying it.

`cd.conformance(graph, stimulus, steps=60)` settles a `NeuralGraph` (`brain.learner.brain`
for a composed brain) from rest and compares its trajectory with the neuron-by-neuron float64
reference. It returns `backend`, `steps`, `max_abs_deviation` over the trajectory, the
reference transmission `ledger` and `final_active` (neurons at activation 0.5 or above).

## Implementation notes

On the CPU backend transport is dense, block products between neuron ranges, when the block
layout fits `NeuralGraph(dense_limit=2048)` (at most `dense_limit ** 2` entries); the CPU paths
reuse the products of a source range (one that receives no synapses) whose activation did not
change. MLX checks each source range for equality on the host at every step, one
synchronization per range per step, and reuses the products of an unchanged one.

Torch keeps a settle's state on its device neuron-major, `(neurons, batch)`, and sums each
sweep's synaptic input by a transport plan made once per graph. The rows of the synapse matrix
(one per postsynaptic neuron, and a second for the apical compartment) that read the same
presynaptic neurons, such as the feature maps of one receptive field, form a group; the groups
of one shape are one batched product over a gather of their presynaptic rows. Nothing is
padded, every row is written by one product in a fixed order, so the device sum is
deterministic. A graph whose rows share few presynaptic neurons uses the dense blocks above
when they fit, one product per block, and a CSR product otherwise (cuSPARSE on CUDA, whose sums
are not deterministic). On CUDA with Triton (bundled with the CUDA builds of PyTorch on Linux)
every group of fewer rows is one Triton launch that accumulates each output in a fixed order,
and the neuron update of a sweep (potential, activation, adaptation, movement) is one fused
launch; `CADENCE_TRITON=0` keeps cuBLAS products and one torch operation at a time. The sweeps
of a settle are captured once as CUDA graphs and replayed (`CADENCE_CUDA_GRAPHS=0` runs them
eagerly; a replay is bitwise the eager sweep), drives and masks reach the device through pinned
host buffers, and a sweep allocates nothing. `CADENCE_TORCH_TRANSPORT=grouped`, `blocks` or
`sparse` forces a plan. Every plan matches the float64 host path to rounding: the same fixed
points and step counts, activations within 1e-10 (`tests/test_torch_settle_parity.py`).

Above `dense_limit` the CPU transport is sparse: a SciPy CSR multiply when SciPy is installed,
a NumPy segmented sum otherwise; `"mlx"` needs the blocks. CSR indices are cached on the connectome (about `4 * (E + N + 1)` bytes for E synapses
and N neurons), and each brain's matrix shares its weights. Summation order differs slightly
between kernels. `graph.to_dict()["transport"]` is `"dense"` or `"segmented"`; `"sparse_kernel"`
is `None` until a sparse CPU multiply runs, then `"scipy_csr"` or `"numpy_segmented"`.

With Numba installed, the CPU backend settles blocked connectomes in one compiled loop (block
transport, then each neuron's update, activation, adaptation and nudge in place) with the
NumPy loop's float64 arithmetic whenever no trajectory or recording is requested, and computes
`residual` the same way; a graph with an apical compartment is settled there too, its pool RMS
summed in NumPy's order. The rows of a batch settle on Numba's threads, as many as
`NUMBA_NUM_THREADS` or `numba.set_num_threads` allow, to the same bits at any count, once a
sweep covers at least 65,536 neurons over its rows (an 8,000-neuron brain from a batch of 8);
a smaller sweep, a single row, or `NUMBA_NUM_THREADS=1` never starts them. A sweep waits for
its slowest row, so on a machine busy with other work the threads can be slower than one: each
worker of a process pool should set `NUMBA_NUM_THREADS=1` (or its share of the cores), since
every process otherwise starts one thread per core. On some machines the compiled loop's last
bits differ from the NumPy loop's, and thousands of lessons can grow that into different
readings (see [the depth benchmark](../benchmarks/depth/README.md)). `CADENCE_FUSED=0` in the
environment before `cadence` is imported forces the NumPy loop, on which the depth receipts
and the look-back, contextual-choice, held-heads, copy-tokens and own-action receipts were
recorded.

A recall memory's reads, writes and address lessons run on the host under every backend: products
of each stream's address or attention with its slots, and the copy of the slots a write makes.
`CADENCE_RECALL_THREADS=N` shares those of at least 262,144 values over blocks of streams,
using at most N workers and at most 32 workers in one reusable pool per process. Canonical
contiguous operands keep the same summation per output; other layouts use the original NumPy
call. The default, 1, dispatches no threaded work. Changing N reuses the bounded pool, and a
forked child starts its own pool. As with Numba's threads, a worker of a process pool should
keep its share of the cores.

Recall may reuse raw content dot products for unchanged addresses and keys. It applies the
current temperature on every read, so retuning and save/load agree. Within one public operation
of the brain (`act`, `lesson`, `step`, `live`, `wait`) the reuse is bound to the bank array
itself: no caller runs between the operation's reads, and the memory replaces its bank rather
than writing into it. Between operations expert key arrays remain writable, so a bitwise bank
snapshot is checked before reuse; this validation costs host memory traffic of the bank's size,
which inside an operation is no longer paid on every read. The cache holds one bank and at most
four addresses per head; a slot write, a full reset and the end of an operation clear it. These
implementation changes do not establish a new task score or a general training-speed ratio.

A device kernel implements five operations: the synaptic transport, the elementwise neuron
update, the softmax over the nudged group, the apical normalization and the largest movement
for the tolerance. Check a new one against the
neuron-by-neuron reference through `conformance`; `tests/test_brain.py` holds the tests every
kernel passes.
