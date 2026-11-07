# Backends, devices, precision

<!-- requires: torch -->

`Brain.compose` takes the fastest backend the machine has for the brain's size and rows
([the default backend](#the-default-backend)); every other graph brain runs on the `"cpu"` backend in
NumPy float64 unless you pick another. Start from the NumPy-only
[installation](../README.md#get-started) and add an extra:

```bash
python -m pip install "cadence-net[fast]==0.94.0"
```

`[fast]` installs Numba and SciPy, `[accel]` PyTorch and `[apple]` MLX; from a checkout, use
`python -m pip install -e ".[fast]"`. `Brain.compose` uses what is installed where it is the fastest, and
everything else takes `backend="torch"` or `backend="mlx"` by name. `available_backends()` lists the
importable backends with the device and default precision each would use.

```python
import cadence as cd

print(cd.available_backends())
# e.g. {'cpu': 'numpy float64', 'torch': 'mps float32', 'mlx': 'gpu float32'} on an M-series Mac
```

## Choose a backend

| backend | where it runs | precision | extra | use it for |
|---|---|---|---|---|
| `"cpu"` | NumPy, with Numba and SciPy when installed | float64 | none; `[fast]` to accelerate | first examples, receipts, numerical checks, sparse graphs, small continuing brains |
| `"torch"` on CUDA | NVIDIA GPU | float32, or float64 with `precision="float64"` | `[accel]` | deep brains, large batches, long lives |
| `"torch"` on MPS | Apple silicon GPU through Metal | float32 only | `[accel]` | a Mac when torch is what you have |
| `"mlx"` | Apple silicon GPU through MLX, unified memory | float32 | `[apple]` | a Mac's wide brains and large batches, on blocked connectomes |
| `"torch"` on CPU | torch on the host | float64, or float32 with `precision="float32"` | `[accel]` | one code path on a machine without a GPU |

`backend="torch"` picks CUDA, then MPS, then CPU; `device=` overrides it. One brain uses one
device. MPS refuses float64, `"mlx"` refuses a connectome whose blocks exceed `dense_limit`,
and a float32 backend refuses a neuron rule whose constants float32 cannot represent.

On MLX a composed brain settles and learns as on the CPU within float32 rounding, lives,
imagines, saves and continues a pending outcome, reads several recall heads, grows and holds an
apical compartment or a learned code ([`tests/test_mlx_backend.py`](../tests/test_mlx_backend.py),
[`tests/test_mlx_launch.py`](../tests/test_mlx_launch.py)).
On a float32 backend a brain reloaded from its checkpoint continues within that rounding, not
bit for bit; on the CPU the reloaded brain is exact.

`available_backends()` asks MLX for its device in a child interpreter, once per process, so an
unsupported Metal or system configuration that aborts MLX's discovery ends only the child. MLX
is then absent from the list, and `backend="mlx"` raises `ImportError` with the child's reason
(`ModuleNotFoundError` when MLX is not installed). The child imports only from the program's
import path, paths added at run time included. An MLX the program has imported, and any MLX in
a frozen application (whose executable is the application itself), answers in the program with
its present default device. Only a found MLX is
remembered, so an MLX installed later in the process is found
([`tests/test_mlx_discovery.py`](../tests/test_mlx_discovery.py)).

## What runs where

`backend=` (`Brain.compose`: default `"auto"`; `NeuralGraph`: default `"cpu"`) and `device=` (default
`None`, the best device) are construction arguments of `NeuralGraph` and `Brain.compose`; `Brain.retune`
refuses them. `precision=` (`"float64"` or `"float32"`, read by torch: float32 on a device, float64 on the
CPU device, unless named) is an argument of `NeuralGraph`, `Brain.compose`, `Brain.load` and `Learner.load`.
Loads default to the saved precision. `Brain.load` with no backend takes the rule's choice again for a
brain the rule placed, and the CPU for any other; `Learner.load` defaults to the saved backend.
`Learner.load(path, backend="cpu")` brings a checkpoint learned on any device back to the
float64 receipt backend.

On CUDA and on MLX a composed brain's memories rest on the device once a moment is large enough to pay
for it (the [rule and its readings](#memories-on-the-device)); below that size, on MPS and a torch CPU
device, they run on the host.

On a device backend `settle_batch` returns a state that stays resident (`BrainState.device`;
`None` on the CPU and from the one-stimulus `settle`), so a continuing phase starts there.
The learning contrast is computed there too (`NeuralGraph.contrast_on_device`, `None` unless
both states are resident; MLX needs a blocked graph). Torch `Learner` updates, momentum and RMS
normalization stay on the device, and so does the apical gain a nudged phase holds
(`NeuralGraph.apical_gain`). On MLX a lesson's update stays on the device from 131,072 block entries,
where every entry holds one synapse, each block has one gain, every tie is one block's entries with
another's, transposed, and every synapse is plastic at one rate: the contrast, the moments and the
normalization by block and over the neurons in float32, the ties, the bound, the new blocks and the
biases in one launch. Below that, MLX hands the contrast to a host optimizer. `ActorCritic` updates on the
device when both of its phases settled there, by the same sizes on MLX. Its dopamine is the host's, and below
the [memories' size](#memories-on-the-device) so is its critic, which reads the critic's columns of a state and
nothing else of it. Reading parameters or saving a checkpoint copies arrays back. Parameter edits and the `learner.brain`
rule are in [contracts](contracts.md).

### Memories on the device

On CUDA from a moment of `streams * width * (width + 1024) >= 3,000,000` and on MLX from `8,000,000`
(`width` the observation code's), the recall memory's slots, maps, read, write and lessons, the working
trace, the consolidating store, the carried routes' contrast, the critic's weights and eligibility and the
moment's drive run on the device in the brain's precision, by the host's equations, the same code on both
([`cadence/resident.py`](../src/cadence/resident.py)): a float64 life agrees with the float64 host path
within rounding at each moment, and a float32 life within float32's
([`tests/test_memories_on_device.py`](../tests/test_memories_on_device.py),
[`tests/test_mlx_memory.py`](../tests/test_mlx_memory.py)). In a float32 brain all of it is float32, the
synapses' eligibility, efficacies and biases and the optimizer's history with it. 3,920
inputs reach that size at one stream, and 35 inputs at about 128 on CUDA and 256 on MLX. Below it each
operation's launch costs more than the host's arithmetic, so the memories stay on the host. On an A10G a brain
of 12 to 1,024 inputs crosses over there, within about 10% on either side, and so does a brain of 12 to 3,920
inputs on an Apple M4. On MLX a moment's reads join the launch that settles on them, and a slot is written in
place in unified memory.

A memory with several heads, a window, `recall_action`, `recall_when`, `recall_gain` or
`recall_tokens`, a store with a separator or a surprise gate, `memory_choice` and a learned code keep
the moment's memories on the host at every size.

Reading one of these arrays on the host (`brain.recall.keys`, `brain.working_memory.trace`,
`brain.hippocampus.strength`, `brain.basal_ganglia.w_critic`), saving a checkpoint, copying the brain,
imagining or retuning the memory
brings them back, and the next moment places them again; nothing else about a brain changes.
`brain.stimulus` returns a host array whichever way the moment runs.
`CADENCE_TORCH_MEMORY=0` keeps them on the host, and `=1` places them at any size on any torch device
but MPS, which is how the tests read this path on a machine without a GPU. `CADENCE_MLX_MEMORY` does the
same on MLX.

Each moment still crosses: to the device the observations, a sampled action's target, the outcome's
reward, key and dopamine; to the host the motor columns, a value per stream, one number per stream and
carried route, one small read per residual check and per settle, and one read of the lesson's or
update's report and finite checks. The memories, the eligibility and the optimizer's state do not
cross ([`tests/test_moment_on_device.py`](../tests/test_moment_on_device.py)).


## The default backend

With no backend named, `Brain.compose` uses `backend="auto"`: the fastest backend this machine has for
the brain's size and rows, in float32 on a device. Rows are known at a brain's first moment, so it composes
on the host and places itself there, with nothing learned yet to move. A brain a device passes at one row
takes the device at compose. Reading `brain.recall` or `brain.brain` before a first moment that moves the
brain holds the host's object; read them after it, or name the backend.

| Device | A brain takes it from | Read on |
| --- | --- | --- |
| CUDA, float32 | `rows * synapses >= 850,000`, under an apical compartment 150,000 | an NVIDIA A10G against the compiled CPU path at its default on its EPYC 7R32, over `act`, `lesson` and `step` |
| MLX, float32 | `rows * synapses >= 500,000`; a brain no launch takes, past 2,048 settling neurons or 1,360 under a compartment, from 32 rows | an Apple M4 against one of its cores |
| The compiled CPU path | every other brain | |

On that A10G, in milliseconds of `act`, `lesson` and `step`, one CPU core at its default solve, then the device:

| Brain | Synapses | Rows | One core | A10G, float32 |
| --- | ---: | ---: | ---: | ---: |
| The chamber's default brain, 35 inputs | 13,120 | 1 | 0.82, 4.4, 4.4 | 2.76, 11.5, 9.2 |
| The same | | 16 | 1.04, 7.9, 9.9 | 3.05, 13.2, 11.5 |
| The same | | 64 | 3.63, 34.4, 33.1 | 3.89, 23.3, 17.6 |
| The same | | 256 | 8.79, 86.0, 87.6 | 4.51, 32.5, 25.7 |
| 128 inputs, `modules=(64,)` | 30,976 | 16 | 1.78, 13.0, 18.8 | 3.49, 18.2, 14.3 |
| The same | | 64 | 6.23, 46.2, 66.2 | 4.60, 22.7, 19.6 |
| 512 inputs, `modules=(128,)` | 222,464 | 4 | 2.77, 22.5, 25.8 | 4.65, 20.1, 18.8 |
| The same | | 16 | 7.80, 36.1, 71.7 | 4.59, 20.0, 18.9 |
| 1,024 inputs, `modules=(256,)` | 870,912 | 1 | 2.94, 59.6, 52.2 | 4.19, 31.5, 25.6 |
| The speech size, 3,920 inputs, `modules=(512,)` | 6,321,152 | 1 | 11.2, 623, 444 | 5.04, 42.6, 28.7 |
| The same | | 16 | 52.2, 642, 1,142 | 6.18, 46.5, 44.6 |
| `modules=(128, 64, 64, 64), apical=0.25, teaching="deep"`, 48 inputs | 50,048 | 1 | 1.63, 7.4, 6.6 | 3.33, 14.4, 12.2 |
| The same | | 4 | 4.20, 20.0, 18.4 | 3.52, 14.7, 12.8 |
| The same | | 64 | 55.0, 170, 119 | 5.20, 32.4, 22.9 |

MPS and other torch devices are not on the list: MPS reads slower than MLX and the CPU, and the others
have no reading; name them to use them. A CUDA card this table has not read takes the device from twice the
work, a stated margin with no calibration in this version. Options that keep the moment's memories on the
host take a device from four times the work, and `precision="float64"` named on CUDA from four times, both
reasoned from the readings above and not read at that size. A learned code stays off torch.

A choice is made once, at the first moment: rows that grow later do not move the brain. `brain.placement` states
the `backend`, `device`, `precision`, `chosen_by` (`"auto"`, `"named"` or `"environment"`), `why`, `refused` and
`pending`. `brain.describe()` and a checkpoint state the same wherever the brain is not on the CPU by name, whose
descriptions and files keep their bytes, and a brain loaded elsewhere reports `saved_on`. A device that refuses the
brain falls to the next fastest, and `refused` names it with its reason: detection never fails a compose. Without the optional packages nothing is imported, and below a device's
work neither torch nor MLX is imported where they are installed.

A new brain's free solves take the fast form on the compiled CPU path alone, so by default the host and a
device agree within the solver's tolerance per moment, and panels by a mean over seeds are the check; under
`learning_fast_settle=False` a float64 device agrees with the float64 host within rounding. A run that must
repeat names its backend. `backend="cpu"` keeps its bits, with `recall=False` or `recall_founder=True` as well.
`CADENCE_BACKEND` names one for a whole run where none is named: `cpu`, `mlx`, `torch`, `torch:cuda` or
`torch:cuda:float64`. This library's tests, benchmarks and examples name theirs.

## Choosing a device

```python
connectome = cd.layered(4, 16, 2, seed=0)
neuron_model = cd.learning_neuron_model()
cd.NeuralGraph(connectome, neuron_model, backend="torch")                        # cuda, else mps, else cpu
cd.NeuralGraph(connectome, neuron_model, backend="torch", device="cpu")          # force a device
cd.NeuralGraph(connectome, neuron_model, backend="torch", precision="float64")   # float64 on a device
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
On an NVIDIA A10G, one process, milliseconds per act and per taught lesson of a newborn, CUDA
against one CPU core of the same machine (an AMD EPYC 7R32) with Numba, the same sweeps:

| Brain | Rows | One CPU core, act and lesson | CUDA float64 | CUDA float32 |
| --- | ---: | ---: | ---: | ---: |
| The chamber's default brain: 251 neurons, 13,120 synapses | 1 | 0.56, 2.76 | 2.64, 10.8 | 2.69, 10.7 |
| The same | 16 | 1.53, 7.81 | 2.95, 13.6 | 2.94, 12.1 |
| The same | 64 | 4.57, 24.2 | 3.42, 25.9 | 3.39, 25.4 |
| The speech size, `Brain.compose(3920, 37, modules=(512,))`: 12,821 neurons, 6.3 million synapses | 1 | 9.46, 393 | 10.0, 236 | 10.0, 238 |
| The same | 16 | 52.9, 493 | 37.1, 304 | 37.3, 296 |
| The same | 64 | 160, 834 | 104, 515 | 103, 483 |
| The same | 256 | 597, 2,270 | 366, 1,332 | 362, 1,263 |
| The same | 1,024 | 2,889, 9,907 | 1,577, 4,828 | 1,602, 4,662 |
| A deep apical brain, `modules=(128, 64, 64, 64), apical=0.25, teaching="deep"`: 535 neurons, 50,048 synapses | 1 | 1.36, 6.30 | 3.54, 14.2 | 3.26, 13.0 |
| The same | 16 | 10.6, 37.3 | 3.86, 15.4 | 3.44, 13.8 |
| The same | 64 | 38.7, 134 | 5.35, 20.7 | 4.42, 16.6 |
| The same | 256 | 152, 524 | 12.8, 44.0 | 8.73, 27.9 |
| The deep anatomy of [`benchmarks/torch_gpu`](../benchmarks/torch_gpu/README.md) under its compartment: 7,956 neurons, 903,520 synapses | 64 | 895, 2,245 | 25.9, 55.4 | 15.1, 26.4 |
| The same | 256 | 3,478, 8,617 | 93.5, 191 | 54.6, 79.5 |

A settle of 32 sweeps from a warm state under a new drive on that device, in milliseconds:

| Brain | Rows | CUDA float64 | CUDA float32 |
| --- | ---: | ---: | ---: |
| The chamber's default brain | 1 | 0.70 | 0.58 |
| The same | 16 | 0.79 | 0.60 |
| The same | 1,024 | 2.49 | 1.10 |
| The speech size | 1 | 1.25 | 0.93 |
| The same | 16 | 3.77 | 1.34 |
| The same | 64 | 9.14 | 2.82 |
| The same | 1,024 | 98.9 | 30.9 |
| The deep apical brain | 16 | 1.55 | 0.90 |
| The same | 256 | 5.84 | 1.38 |
| The deep anatomy | 16 | 6.13 | 2.39 |
| The same | 1,024 | 209 | 49.6 |

A settle is a small part of an act of a brain with the memory, whose reads run on the host. The
device's float64 runs at a sixty-fourth of its float32 rate; check float32 against float64 as below
before relying on it. On the small continuing nursery of the
[System 1 CUDA qualification](../benchmarks/system1_cuda/RESULTS.md), CPU with Numba is faster
than CUDA.

On an Apple M4, one process, milliseconds per act and per taught lesson of a newborn, MLX
against one CPU core with Numba (float32 against float64, the same sweeps), at a load of about 2.5:

| Brain | Rows | One CPU core, act and lesson | MLX, act and lesson |
| --- | ---: | ---: | ---: |
| A small brain without the memory: 12 inputs, association 32 | 1 | 0.07, 0.26 | 0.66, 2.8 |
| The same with the memory a new brain holds | 1 | 0.17, 0.77 | 0.73, 3.1 |
| The chamber's recipe: 251 neurons, 13,120 synapses | 16 | 0.53, 2.8 | 0.67, 3.1 |
| A new brain at the speech size: 3,920 inputs, 6.3 million synapses, 24 million parameters | 1 | 14, 291 | 2.8, 23 |
| The same | 16 | 17, 295 | 6.2, 34 |
| The same | 64 | 49, 411 | 17, 70 |
| The same | 256 | 185, 922 | 60, 205 |
| The same | 1,024 | 751, 3,020 | 347, 1,260 |
| The same | 4,096 | 3,230, 19,500 | 1,520, 7,250 |
| Four regions under an apical compartment, `modules=(128, 64, 64, 64)`, without the memory: 439 neurons, 43,904 synapses | 1 | 0.42, 1.8 | 0.73, 3.3 |
| The same | 16 | 1.9, 5.5 | 0.79, 3.3 |
| The same | 64 | 8.4, 23 | 5.9, 21 |
| The same with the memory: 535 neurons, 50,048 synapses | 1 | 0.50, 2.5 | 0.85, 4.5 |
| The same | 16 | 2.4, 9.3 | 1.1, 5.6 |
| The same | 64 | 13, 55 | 3.9, 13 |
| The speech size under a compartment, `modules=(256, 128)`, with the memory: 12,309 neurons, 2.1 million synapses | 1 | 27, 191 | 7.2, 30 |
| The same | 16 | 101, 441 | 9.6, 40 |
| The deep anatomy above, with its compartment: 7,956 neurons, 903,520 synapses | 1 | 33, 118 | 99, 260 |
| The same | 16 | 132, 426 | 106, 537 |
| The same | 64 | 248, 650 | 118, 350 |
| The same | 256 | 1,090, 2,680 | 312, 896 |
| The same | 1,024 | 4,550, 11,400 | 1,370, 3,560 |
| The same | 4,096 | 18,000, 49,100 | 7,130, 25,500 |

MLX wins at the speech size at every batch, on four regions under a compartment from about 16 rows, and on the
deep anatomy, which takes the compiled sweep, from about 32 rows. At the chamber's size one synchronization with the
GPU costs about 180 microseconds whatever the work, and a solve of three chunks takes three: a pass of the
[grammar chamber](../benchmarks/grammar/README.md), 4,000 batches of 16 rows, takes 51 s on one CPU core and
75 s on MLX. One core stays the choice at that size. The load moves these readings: above a load of 4 an MLX act
at the speech size has read twice its quiet time.

Cap BLAS and Numba threads for parallel workers. To compare two implementations, use identical
dtype, steps and tolerances over several seeds, warm up, alternate the order, synchronize
around each timed call, and compare residuals and states too.

## Precision matters

Near multiple attractors a rounding difference can change the selected state, and float32
learning need not track float64 within a fixed error; RMS normalization amplifies small
contrast errors. Every backend forms the phase contrast as
`A_plus.T @ (B_plus - B_minus) + (A_plus - A_minus).T @ B_minus`, so identical phases give
exactly zero. Check the deployed backend with `conformance` on representative drives, compare
residuals and task readouts, and declare precision in any receipt. Run gradient checks on
float64 CPU or CUDA with the nudge size and phase residuals declared.

A deep brain under an apical compartment shows what this means. One lesson on MLX or on torch at
float32 reads a synapse's contrast within 1e-8 of the float64 reading, where the contrasts of the
deepest projection stand near 1e-7 at birth and 4e-6 once it learns. Its step agrees within 3e-5
wherever the contrast reaches 1e-6, and below that rounding sets the sign. Those differences do not
stay small: from one checkpoint on the same rows the efficacies of float64 and float32 part from 4e-7
after one lesson to 4e-3 after 200, and two float64 runs part the same way from a start rounded to
float32, reading .72 and .52 at lesson 1,200. So a float32 run is another run. What it learned reads
the same on every backend: the same answer on all 2,000 held rows of
[`benchmarks/depth/and_count.py`](../benchmarks/depth/README.md) on the CPU, on MLX and on torch at
float32, whichever of the first two taught it. On that benchmark's four regions over 4,000 lessons,
the mean of the ten readings from lesson 3,100 over six seeds is .802 on the CPU (.719 to .831) and
.786 on MLX (.751 to .817): MLX reads .016 below, with a standard error of .015 on that difference. A
single reading swings further on both, down to .40 past lesson 2,000 on the CPU and .39 on MLX, so
compare runs by such a mean over seeds, not by a last reading.

## Residual checks, masks and conformance

`NeuralGraph.equilibrate` chooses its work from the potential and adaptation equation
residual: a warm state within tolerance takes zero steps and one residual check. `chunk` trades
check frequency against overshoot at the same final tolerance. Step count measures numerical
settling work, never surprise or task quality. `residual(drive, state, on_device=True)` checks
an unread torch state on its device in float64, one scalar per row back (a float32 state
through a float64 copy of the kernel, the same float64 reference); a state an MLX launch made reads
that launch's own check, in float32, under the same drive, nudge and parameters; other MLX states,
MPS states, a state read on the host and `on_device=False` use the float64 host reference. A solve
on MLX decides by the launch's check from a tolerance of 1e-4, where its rounding is about a
hundredth of the tolerance, and by the reference below that. A positive movement
`tolerance` in `settle_batch` tests every step; on CUDA the test is read one sweep behind (the
next sweep is queued before the host reads the flag, into the other state slot), so the host
never stalls the device, and the reported state and step count are those of the first sweep
that met the tolerance. An MLX launch reads which of its sweeps moved a neuron and, where it went
past the stop, makes the state again from its start: the same steps and state, the extra sweeps in
`discarded_steps` and `attempted_steps`. Fixed step counts and the inner chunks of `equilibrate`
read nothing.
CPU and MPS check before starting another sweep. For a direct CUDA movement-tolerance solve,
`BrainState.steps` is the returned state's sweep index, `discarded_steps` counts lookahead,
and `attempted_steps` counts both. Use `attempted_steps` for executed-work comparisons; it counts
sweeps and measures no total resource or energy.

When arousal changes the sampling heat, torch workspaces reuse their device temperature inputs
and captured graphs read the values in place. Topology and arithmetic-branch changes use
bounded caches that evict dependent configurations and captures together, so a life with a
changing temperature does not accumulate nudge tensors.

Masks have shape `(n,)` or `(1, n)` for every row, or `(batch, n)` per row, with entries in
[0, 1], on every backend. A mask scales potential and activation at each step; zero removes a
neuron's published activity while its adaptation decays. A warm start recomputes the published
activation under the current mask; the fused kernel is handed a shared mask laid out per row,
and no mask at all when none is given.

`cd.conformance(graph, stimulus, steps=60)` settles a `NeuralGraph` (`brain.learner.brain`
for a composed brain) from rest and compares its trajectory with the neuron-by-neuron float64
reference. It returns `backend`, `steps`, `max_abs_deviation` over the trajectory, the
reference transmission `ledger` and `final_active` (neurons at activation 0.5 or above).

## Implementation notes

On the CPU backend transport is dense, block products between neuron ranges, when the block
layout fits `NeuralGraph(dense_limit=2048)` (at most `dense_limit ** 2` entries); the CPU paths
reuse the products of a source range (one that receives no synapses) whose activation did not
change.

MLX runs a chunk of sweeps in one Metal launch (`mx.fast.metal_kernel`): one threadgroup per row,
one thread per settling neuron (a neuron of a range that receives synapses), the activations in
threadgroup memory, a barrier per sweep, and the equations' residual of the launch's first and
last state read at its end, so a solve that needs one chunk synchronizes with the device once.
Without adaptation a source range's potential follows its drive in closed form. Under `dt = 1` it is
its drive from its first sweep on, so its products are made once a settle, one MLX operation each,
and reach the launch summed. Under another step each sweep reads the products of its own, made in one
compiled operation, until the range's gap to its drive is below 1e-7: about 22 sweeps after a new
observation at `dt = 0.5`. With an apical compartment a thread reads a neuron's basal and top-down
products apart, a second barrier per sweep lets it read its pool's mean squares, and it adds the
top-down input at the pool's gain; a nudged settle reads the gains it holds. A launch's last state
leaves its gains, so `apical_gain` after a free settle takes no launch of its own. The
state stays on the device; the host's arrays are made from it without another launch when read.
A launch holds up to 2,048 settling neurons, 1,360 with a compartment, and takes no mask, anchor,
trajectory or recording, no block holding synapses of both compartments, no adaptation, no softmax
group that is not a range of settling neurons and no temperature below 0.02. Those settles run a
sweep through `mx.compile`, the blocks as inputs, with every block's
product at every step, and evaluate a settle's graph once at its end (at least every 64 steps of a
long one), or at every step when a movement `tolerance` is read. `CADENCE_MLX_LAUNCH=0` keeps one MLX
operation at a time and the update and the maps on the host.

Torch keeps a settle's state on its device neuron-major, `(neurons, batch)`, and sums each
sweep's synaptic input by a transport plan made once per graph. The rows of the synapse matrix
(one per postsynaptic neuron, and a second for the apical compartment) that read the same
presynaptic neurons, such as the feature maps of one receptive field, form a group; the groups
of one shape are one batched product over a gather of their presynaptic rows. Nothing is
padded, every row is written by one product in a fixed order, so the device sum is
deterministic. A graph whose rows share few presynaptic neurons uses the dense blocks above
when they fit, one product per block, and a CSR product otherwise (cuSPARSE on CUDA, whose sums
are not deterministic). `CADENCE_TORCH_TRANSPORT=grouped`, `blocks` or `sparse` forces a plan.
Every plan matches the float64 host path to rounding: the same fixed points and step counts,
activations within 1e-10 (`tests/test_torch_settle_parity.py`).

Under a step of one, without adaptation or an apical compartment, on CUDA or the torch CPU device,
the products of source neurons (those no synapse reaches: the sensory, prefrontal and recall ranges
of a composed brain) are a part of the plan of their own, 99.4% of the entries of the speech-size
brain below and 82% of the chamber's. Such a neuron stands at its drive from the second sweep of a
settle. Sweeps 0 and 1 make those products, later sweeps read them, and from sweep 3 the fused update
covers only the neurons synapses reach. A mask, or a nudge that reaches a source neuron, makes every
product at every sweep. From 4,194,304 entries times rows the workspace keeps them with the source
potentials they were made from, so a settle that starts from those potentials, or under a drive that
holds them, and a residual check of such a state make none, under the same parameters.
`CADENCE_STILL_PRODUCTS=0` makes every product at every sweep.

On CUDA with Triton (bundled with the CUDA builds of PyTorch on Linux) the rest are launches where
a measurement favours them. Several groups of one shape always are, and a group of up to 128
presynaptic neurons at float64 or 256 at float32: a launch costs about 5 microseconds there and a
cuBLAS product about 25 at float64 and 5 at float32. At float64 from a batch of 8 a group of up to
4,096 presynaptic neurons is too, and a larger one once it fills about 64 programs. cuBLAS keeps the
others. A program computes a tile of output rows over a tile of the batch, 16 presynaptic neurons at a
time at float64 and 32 at float32, and accumulates each output in a fixed order; a group takes the
tile height its rows fill, and a batch the tile it fills, since a program's mask does not spare its
arithmetic. The neuron update of a sweep (potential, activation, adaptation, movement) is one fused
launch, and the apical normalization three: the sums of squares over tiles of a pool's members, each
pool's gain, the drive. `CADENCE_TRITON=0` keeps cuBLAS products and one torch operation at a time.
The sweeps of a settle are captured once as CUDA graphs and replayed (`CADENCE_CUDA_GRAPHS=0` runs
them eagerly; a replay is bitwise the eager sweep), drives and masks reach the device through pinned
host buffers, and a sweep allocates nothing.

Above `dense_limit` the CPU transport is sparse: a SciPy CSR multiply when SciPy is installed,
a NumPy segmented sum otherwise; `"mlx"` needs the blocks. CSR indices are cached on the connectome (about `4 * (E + N + 1)` bytes for E synapses
and N neurons), and each brain's matrix shares its weights. Summation order differs slightly
between kernels. A brain whose regions are dense pays for this: at 3,920 inputs a recall
brain's blocks exceed 2048 squared entries, and on the sparse transport an act takes about 30 times
as long. `Brain.compose` with `recall=True` therefore takes a limit as wide as the blocks wherever at
least half of their entries are synapses, which costs 8 bytes per block entry, no more than the
sparse form; `dense_limit=` names another. A brain without the memory composes at 2048. `graph.to_dict()["transport"]` is `"dense"` or `"segmented"`; `"sparse_kernel"`
is `None` until a sparse CPU multiply runs, then `"scipy_csr"` or `"numpy_segmented"`.

With Numba installed, the CPU backend settles blocked connectomes in one compiled loop (block
transport, then each neuron's update, activation, adaptation and nudge in place) with the
NumPy loop's float64 arithmetic whenever no trajectory or recording is requested, and computes
`residual` the same way; a graph with an apical compartment is settled there too, its pool RMS
summed in NumPy's order. A state these kernels settled or checked keeps the block products of
its activations: a check of the state, or a settle from it, makes only the products of the
ranges whose activations differ from the ones they were made from, to the same bits (a bounded
solve's check makes those of the ranges the chunk's last sweep moved, and the next chunk's
first sweep makes none). The state keeps its potentials and activations too, so a launch whose
potentials are the kept ones reads their activations without computing them. The products hold
for the blocks they were made with; `copy` and `pickle` leave them out. A free solve without a
mask, a nudge or a compartment runs its check and its chunks in one compiled call, unless
`settle_batch` or `residual` is wrapped, which then sees every launch. The kernels compile at their
first use under a source version, about 10 seconds on an Apple M4, and later processes load them
from Numba's cache (`NUMBA_CACHE_DIR` where the package's folder is read-only). The rows of a batch settle
on Numba's threads, as many as
`NUMBA_NUM_THREADS` or `numba.set_num_threads` allow, to the same bits at any count, once a
sweep covers at least 65,536 neurons over its rows (an 8,000-neuron brain from a batch of 8);
a smaller sweep, a single row, or `NUMBA_NUM_THREADS=1` never starts them. A sweep waits for
its slowest row, so on a machine busy with other work the threads can be slower than one: each
worker of a process pool should set `NUMBA_NUM_THREADS=1` (or its share of the cores), since
every process otherwise starts one thread per core. On some machines the compiled loop's last
bits differ from the NumPy loop's, and thousands of lessons can grow that into different
readings (see [the depth benchmark](../benchmarks/depth/README.md)). `CADENCE_FUSED=0` in the
environment before `cadence` is imported forces the NumPy loop; use it to reproduce the depth,
look-back, contextual-choice, held-heads, copy-tokens and own-action receipts, which were
recorded on that loop.

On MLX a recall address map of at least 1,048,576 entries of a memory that keeps to the host (several
heads, a window, the command) rests on the device: its product with a moment and its step are launched
there, in float32, and it reads as a float64 array wherever an array is asked for. The read back of a
lesson's movement uses the blocks when the synapses rest there.
On the host a recall memory's reads, writes and address lessons are products
of each stream's address or attention with its slots, and the copy of the slots a write makes.
`CADENCE_RECALL_THREADS=N` shares those of at least 262,144 values over blocks of streams,
using at most N workers and at most 32 workers in one reusable pool per process. Canonical
contiguous operands keep the same summation per output; other layouts use the plain NumPy
call. The default, 1, dispatches no threaded work. Changing N reuses the bounded pool, and a
forked child starts its own pool. As with Numba's threads, a worker of a process pool should
keep its share of the cores.

Recall reuses raw content dot products for unchanged addresses and keys and applies the
temperature on every read, so retuning and save/load agree. Within one public operation of the
brain (`act`, `lesson`, `step`, `live`, `wait`) the reuse is bound to the bank array itself: no
caller runs between the operation's reads, and the memory replaces its bank rather than writing
into it. Between operations expert key arrays are writable, so a bitwise bank snapshot is
checked before reuse, at a cost of host memory traffic of the bank's size. The cache holds one
bank and at most four addresses per head; a slot write, a full reset and the end of an
operation clear it.

A device kernel implements five operations: the synaptic transport, the elementwise neuron
update, the softmax over the nudged group, the apical normalization and the largest movement
for the tolerance. Check a new one against the
neuron-by-neuron reference through `conformance`; `tests/test_brain.py` holds the tests every
kernel passes.

### Choosing kernels by measurement

`graph.to_dict()["torch_transport"]` reports what a torch graph runs: its kind, its products and
launches at a batch of 16, how many of them are still products, and the plan it follows, `"rules"`
until something asks otherwise. `graph.calibrate_transport(batches=(16, 64))` times, at each batch and
on this device, every bucket in a launch and on cuBLAS, then each part of a sweep under that choice,
under every bucket in the launch, under none and under the rules, and keeps what reads fastest; a
choice replaces the rules only when it reads at least 2% faster. The plan names a launch per bucket
and batch, and a batch it does not name takes the nearest it names. It is a stored object: no later
timing moves it, nothing changes it in the middle of a life, and it is kept per graph shape, device
and precision, in the process and in `CADENCE_PLAN_CACHE` (by default
`~/.cache/cadence/transport_plans.json`), so a second brain of one shape pays nothing. It is repeated
by name with `graph.calibrate_transport(name=plan["name"])`, which refuses a plan made for another
shape, device or precision, and `name="rules"` returns to the rules. A graph on new parameters keeps
its plan; a checkpoint does not carry one. On an A10G a calibration costs 10 to 950 milliseconds per
batch size. On the brains of the table above, at batches 1 to 1,024, it matches the rules at 51 of 60
readings and gains 1.29 at most, at one row. On eight graphs read after the rules were set the rules'
transport stands within 1.86 of the better of every bucket in a launch or none, at one row, and the
plan within 1.06. A device without Triton has no launch to choose, and its plan names none.
