# Backends, devices, precision

<!-- requires: torch -->

`Brain.compose` takes the fastest backend the machine has for the brain's size and rows
([the default backend](#the-default-backend)); every other graph brain runs on the `"cpu"` backend in
NumPy float64 unless you pick another. Start from the NumPy-only
[installation](../README.md#get-started) and add an extra:

```bash
python -m pip install "cadence-net[fast]==0.99.0"
```

`[fast]` installs Numba and SciPy, `[accel]` PyTorch and `[apple]` MLX (0.21.1 or later); from a checkout, use
`python -m pip install -e ".[fast]"`. `Brain.compose` uses what is installed where it is the fastest, and
everything else takes `backend="torch"` or `backend="mlx"` by name. `available_backends()` lists the
importable backends with the device and default precision each would use.

```python
import cadence as cd

print(cd.available_backends())
# e.g. {'cpu': 'numpy float64', 'torch': 'mps float32', 'mlx': 'gpu float32'} on an M-series Mac
```

## Which backend should I use?

A backend is where a brain's arithmetic runs: NumPy on the CPU, PyTorch on a CUDA GPU or the CPU,
or MLX on an Apple silicon GPU. The answer is usually the default: a new brain takes the fastest
backend for its size at its first moment, by a rule measured on real machines.

| Your situation | Do this | What you get |
| --- | --- | --- |
| Starting out, on any machine | nothing (`backend="auto"`) | small brains on the CPU, large ones on a GPU where one is installed and pays ([the rule](#the-default-backend)) |
| Results that must repeat to the bit: receipts, numerical checks, tests | `backend="cpu"` | NumPy float64, the same bits on every run |
| A Mac with Apple silicon, a wide brain or many streams | install `[apple]`; the default takes MLX past its size, or name `backend="mlx"` | float32 on the GPU |
| An NVIDIA GPU | install `[accel]`; the default takes CUDA past its size, or name `backend="torch"` | float32 on the GPU |
| float64 on a GPU | `backend="torch", precision="float64"` | slower: CUDA's float64 runs at a fraction of its float32 rate |
| One backend for a whole run, without changing code | the environment variable `CADENCE_BACKEND=cpu` (or `mlx`, `torch`, `torch:cuda`, `torch:cuda:float64`) | every brain that names no backend takes it |
| A brain trained on a GPU, read on a CPU | `Brain.load(path, backend="cpu")` | the same brain in float64 |

One minimal example per choice:

```python
import numpy as np

auto = cd.Brain.compose(8, 3)                                   # the default, "auto"
auto.act(np.eye(8)[[0]])                                        # it chooses at its first moment
print(auto.placement["backend"], auto.placement["precision"], auto.placement["why"])
exact = cd.Brain.compose(8, 3, backend="cpu")                   # NumPy float64, repeatable
on_torch = cd.Brain.compose(8, 3, backend="torch", device="cpu")  # torch; device="cuda" for a GPU
if "mlx" in cd.available_backends():
    apple = cd.Brain.compose(8, 3, backend="mlx")               # an Apple silicon GPU
```

`brain.placement` says where a brain runs and why. A float32 brain on a device learns another run
than a float64 one on the CPU: compare them by means over seeds
([precision matters](#precision-matters)).

<a id="choose-a-backend"></a>

### The backends

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

On MLX a composed brain settles and learns by the CPU's equations in float32, within float32 rounding of an
ordinary solve and within the solver's tolerance of [the fast form](#the-default-backend) the CPU takes by default. It
lives, imagines, saves and continues a pending outcome, reads several recall heads, grows, grafts and retunes, and holds
an apical compartment or a learned code ([`tests/test_mlx_backend.py`](../tests/test_mlx_backend.py),
[`tests/test_mlx_launch.py`](../tests/test_mlx_launch.py), [`tests/test_mlx_memory.py`](../tests/test_mlx_memory.py)).
On a float32 backend a brain reloaded from its checkpoint continues within that rounding, not
bit for bit; on the CPU the reloaded brain is exact.

<a id="a-reloaded-brain-and-blas-threads"></a>
On the NumPy path (`CADENCE_FUSED=0`) a reloaded brain's continued life repeats to the bit at
one BLAS thread count. In a reading of 35 compositions on an x86-64 machine, four continued after
a load to other last bits with the thread variables unset than at one thread, where the life
before the save read the same at either count, and each count repeated its own bits (#315). For
a reloaded life that repeats to the bit on another run or machine, set
`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1` before NumPy is imported. The fused
path read one set of bits at either count.

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
another's, transposed, no projection is [tied](tune.md#tied-projections-opt-in) and every synapse is plastic: the contrast, the moments and the
normalization by block and over the neurons in float32, each synapse's rate, the ties, the bound, the new
blocks and the biases in one launch. `brain.learner.synapse_rate = rates` changes a rate there; the array an
update read is read-only, since comparing it per lesson would cost a fifth of a lesson at the speech size. Below that, MLX hands the contrast to a host optimizer. `ActorCritic` updates on the
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

Some options keep the moment's memories on the host at every size. The device form in
[`cadence/resident.py`](../src/cadence/resident.py) holds one read port over one bank of slots keyed on one
event, and each of these adds a part that form does not hold, so the whole moment's memories stay with it on the
host, where the same equations run in NumPy:

| Option | What it adds that the device form does not hold |
| --- | --- |
| `recall_heads=2` or more | several read ports over the one bank, each reading in turn from the share of every slot the heads before it left, each with its own address map |
| `recall_window=k` | a per-stream window of the last k events that the key and the address read in order |
| `recall_action=True` | a bank of the executed commands beside the slots, written from the host's action choice |
| `recall_when=True` | the read's age features, delivered on a region of their own |
| `recall_tokens`, `recall_gain` | the tokens' identity map into the motor region and one learned gain per port, taught by the motor's movement |
| a store with a separator, a surprise gate or `memory_fade` | the separator's expanded code of each key, the gate's noise over the consolidated strengths, or the fade's depth and recent use per key coordinate, read at every moment |
| `recall_cortex_norm=True` | the cortex of the moment before set to unit norm per stream before the maps read it |
| `memory_choice=True` | the episodic records: place and context records, their vote in the choice and the replay of a path |
| `learned_code=units` | a region that settles to its code before the moment's settle, with its own lessons |

These brains take a device from four times the work ([the default backend](#the-default-backend)), so their
crossings count in the choice. On MLX a `maturity` reads the efficacies on the host at each outcome, since it
measures the step applied.

Reading one of these arrays on the host (`brain.recall.keys`, `brain.working_memory.trace`,
`brain.hippocampus.strength`, `brain.basal_ganglia.w_critic`), saving a checkpoint, copying the brain,
imagining or retuning the memory
brings them back, and the next moment places them again; nothing else about a brain changes.
`brain.stimulus` returns a host array whichever way the moment runs.
`CADENCE_TORCH_MEMORY=0` keeps them on the host, and `=1` places them at any size on any torch device
but MPS, which is how the tests read this path on a machine without a GPU. `CADENCE_MLX_MEMORY` does the
same on MLX.

Each moment crosses: to the device the observations, a sampled action's target, the outcome's
reward and key; to the host the motor columns, a value per stream, one number per stream and
carried route, one small read per residual check and per settle, and one read of the lesson's or
update's report and finite checks. On torch the dopamine is made on the device from the value the
critic makes there, and an outcome reads its checks once before anything is installed: the dopamine's,
the critic's, the recall memory's reward step and the update's, with the critic's bias step, the
report's steps and the dopamine of every row, which the carried routes' gains take on the host. On MLX
the dopamine is the host's and goes to the device. The memories, the eligibility and the optimizer's
state do not cross ([`tests/test_moment_on_device.py`](../tests/test_moment_on_device.py)).
[`benchmarks/device/ledger.py`](../benchmarks/device/ledger.py) lists every crossing of a moment by the place
in the library that made it, with its bytes and time, and [its readings](../benchmarks/device/README.md) on an
A10G and an M4 give what each moment pays: its bytes are small, and its synchronizations are what it pays for.

A stream's synapse eligibility takes one float per synapse, 25 MB at the speech size in float32, so `step`
takes its streams from the device's memory: 256 of them fit an A10G's 22 GB in float32 and in float64, and 22 GB
hold about 800 of them in float32 by that arithmetic. On a 24 GB Apple M4 256 of them peak at 7.9 GB, and the 17 GB
Metal gives a process hold about 680 by that arithmetic; past that Metal raises at allocation. `act` and `lesson`
hold no eligibility. A large read eligibility goes to a checkpoint as the actor keeps it, and the reader takes that form
or the full layout.


## The default backend

With no backend named, `Brain.compose` uses `backend="auto"`: the fastest backend this machine has for
the brain's size and rows, in float32 on a device. Rows are known at a brain's first moment, so it composes
on the host and places itself there, with nothing learned yet to move. A brain a device passes at one row
takes the device at compose. Reading `brain.recall` or `brain.brain` before a first moment that moves the
brain holds the host's object; read them after it, or name the backend.

| Device | A brain takes it from, `rows * synapses` | Under an apical compartment | Read on |
| --- | ---: | ---: | --- |
| CUDA, float32 | 800,000 | 110,000 | an NVIDIA A10G against one core of its EPYC 7R32 at the CPU path's default |
| MLX, float32 | 800,000 | 500,000 | an Apple M4 against one of its cores at the CPU path's default |
| The compiled CPU path | every other brain | every other brain | |

On MLX a brain no launch takes, past 2,048 settling neurons or 1,360 under a compartment, takes the device from 32
rows at any size. The crossing is where `act`, `lesson` and `step` together pass the core, by their geometric
mean. On each device one brain near it reads a small loss: 512 inputs at 4 rows, because no single number
also takes 1,024 inputs at one row, which the device runs in half the time.

On that A10G, milliseconds of `act`, `lesson` and `step`, each cell read in turn with the GPU to itself:

| Brain | Synapses | Rows | One core | A10G, float32 |
| --- | ---: | ---: | ---: | ---: |
| The chamber's default brain, 35 inputs | 13,120 | 1 | 0.52, 2.66, 2.73 | 1.61, 8.04, 7.32 |
| The same | | 16 | 0.93, 6.82, 8.36 | 1.99, 9.49, 9.65 |
| The same | | 64 | 2.10, 23.1, 23.9 | 2.72, 17.0, 15.7 |
| The same | | 256 | 8.32, 77.3, 79.9 | 3.49, 16.2, 24.5 |
| 128 inputs, `modules=(64,)` | 30,976 | 16 | 1.37, 11.4, 15.9 | 2.32, 14.1, 12.5 |
| The same | | 64 | 4.25, 31.6, 43.7 | 3.36, 15.4, 18.0 |
| 512 inputs, `modules=(128,)` | 222,464 | 4 | 2.01, 16.6, 22.1 | 3.49, 15.5, 16.3 |
| The same | | 16 | 4.67, 27.9, 52.0 | 3.36, 15.5, 16.7 |
| 1,024 inputs, `modules=(256,)` | 870,912 | 1 | 1.95, 41.8, 31.9 | 2.76, 25.7, 22.3 |
| The speech size, 3,920 inputs, `modules=(512,)` | 6,321,152 | 1 | 7.05, 480, 291 | 3.83, 21.8, 20.9 |
| The same | | 16 | 42.7, 591, 997 | 4.58, 24.3, 35.4 |
| `modules=(128, 64, 64, 64), apical=0.25, teaching="deep"`, 48 inputs | 50,048 | 1 | 1.46, 6.55, 5.35 | 1.91, 9.60, 9.50 |
| The same | | 4 | 3.75, 13.6, 12.7 | 2.02, 10.2, 10.3 |
| The same | | 16 | 10.8, 37.3, 32.7 | 2.24, 11.6, 12.0 |
| `modules=(256, 128), apical=0.25, teaching="deep"`, 3,920 inputs | 2,098,432 | 1 | 40.1, 278, 138 | 4.30, 19.7, 19.3 |
| The same | | 16 | 230, 932, 817 | 5.84, 24.9, 27.6 |

On an Apple M4, the same, single readings on a shared machine at a load of 12 to 25:

| Brain | Synapses | Rows | One core | Apple M4, MLX |
| --- | ---: | ---: | ---: | ---: |
| The chamber's default brain, 35 inputs | 13,120 | 1 | 1.04, 4.51, 5.66 | 3.56, 12.6, 14.4 |
| The same | | 16 | 2.44, 13.3, 12.6 | 6.34, 14.1, 17.2 |
| The same | | 64 | 6.11, 32.9, 36.9 | 7.16, 26.1, 35.3 |
| The same | | 256 | 22.5, 158, 114 | 9.39, 29.6, 102 |
| 128 inputs, `modules=(64,)` | 30,976 | 16 | 4.69, 22.4, 18.1 | 3.24, 16.4, 25.5 |
| The same | | 64 | 10.2, 39.5, 45.6 | 5.92, 16.8, 31.3 |
| 512 inputs, `modules=(128,)` | 222,464 | 4 | 2.98, 24.6, 20.2 | 5.32, 25.9, 25.6 |
| The same | | 16 | 8.51, 63.2, 118 | 5.86, 21.7, 31.9 |
| 1,024 inputs, `modules=(256,)` | 870,912 | 1 | 11.0, 232, 125 | 8.38, 34.6, 28.0 |
| The speech size, 3,920 inputs, `modules=(512,)` | 6,321,152 | 1 | 25.6, 648, 365 | 8.05, 59.2, 44.5 |
| The same | | 16 | 51.6, 849, 2,115 | 15.4, 60.4, 145 |
| `modules=(128, 64, 64, 64), apical=0.25, teaching="deep"`, 48 inputs | 50,048 | 1 | 3.29, 14.5, 13.1 | 3.92, 18.4, 15.1 |
| The same | | 4 | 5.81, 16.5, 14.0 | 5.86, 21.0, 17.3 |
| The same | | 64 | 34.7, 101, 94.9 | 11.6, 36.4, 71.1 |
| `modules=(256, 128), apical=0.25, teaching="deep"`, 3,920 inputs | 2,098,432 | 1 | 73.1, 361, 221 | 10.4, 45.9, 37.8 |
| The same | | 16 | 210, 938, 683 | 13.3, 89.2, 74.7 |

MPS and other torch devices are not on the list: MPS reads slower than MLX and the CPU, and the others
have no reading; name them to use them. Options that keep the moment's memories on the
host take a device from four times the work, and `precision="float64"` named on CUDA from four times, both
reasoned from the readings above and not read at that size. A learned code stays off torch.

A CUDA card this table has not read is measured once. The first brain whose work reaches a quarter of
the table's crossing times the chamber's default brain on the card in float32 and on one core of the
machine: a greedy `act`, a `lesson` and a `step` at 16, 64 and 256 rows, two of each after one to warm.
Where the two cross, in `rows * synapses`, sets the card's crossings: the table's, scaled by that crossing over
the A10G's, which the same reading of the A10G table puts at 573,568 (43.7 rows of the chamber's brain). A card
half as fast as the A10G on that brain takes the device from twice the table's work, and a card twice as fast from
half of it. The calibration composes six brains and runs 72 moments, 54 of them timed. On an A10G it took
2.6 seconds and read the crossing at 43.8 and at 44.3 rows in two runs, within 1.4% of the table's. On an NVIDIA
L4, a card the table has not read, it read 59.8, 53.6 and 50.4 rows in three runs, 2.2 seconds each after the first,
whose 8.0 seconds held the device's start; the cached reading puts the L4's crossings at 1.15 times the table's, so
the chamber's brain takes the CPU at 64 rows and the L4 at 256 there.
Its result is kept beside the [stored transport plans](#choosing-kernels-by-measurement), in
`CADENCE_PLAN_CACHE` or `~/.cache/cadence/transport_plans.json` under `card <name>`, so each card is calibrated
once per machine. `cadence.resident.calibrate_card()` runs it by hand and returns the readings, the crossing
and the seconds it took. `CADENCE_CALIBRATE=0` skips it, and a card without a calibration, or one whose
calibration fails, takes the device from twice the table's work, a stated margin
([`tests/test_default_backend.py`](../tests/test_default_backend.py)).

A choice is made once, at the first moment: rows that grow later do not move the brain. `brain.placement` states
the `backend`, `device`, `precision`, `chosen_by` (`"auto"`, `"named"` or `"environment"`), `why`, `refused` and
`pending`. `brain.describe()` and a checkpoint state the same wherever the brain is not on the CPU by name, whose
descriptions and files keep their bytes, and a brain loaded elsewhere reports `saved_on`. A device that refuses the
brain falls to the next fastest, and `refused` names it with its reason: detection never fails a compose. Without the optional packages nothing is imported, and below a device's
work neither torch nor MLX is imported where they are installed.

A new brain's free solves take the fast form on the compiled CPU path alone, so by default the host and a
device agree within the solver's tolerance per moment, and panels by a mean over seeds are the check; under
`learning_fast_settle=False` a float64 device agrees with the float64 host within rounding. The default brain's
chamber and battery panels were read on the CPU, and `auto` keeps those sizes on the CPU at 16 rows. A run that must
repeat names its backend. `backend="cpu"` keeps its bits, with `recall=False` or `recall_founder=True` as well.
`CADENCE_BACKEND` names one for a whole run where none is named: `cpu`, `mlx`, `torch`, `torch:cuda` or
`torch:cuda:float64`. This library's tests, benchmarks and examples name theirs.

## Choosing a device

```python
connectome = cd.layered(4, 16, 2, seed=0)
neuron_model = cd.learning_neuron_model()
cd.NeuralGraph(connectome, neuron_model, backend="torch")                        # cuda, else mps, else cpu
cd.NeuralGraph(connectome, neuron_model, backend="torch", device="cpu")          # force a device
cd.NeuralGraph(connectome, neuron_model, backend="torch", precision="float32")   # a named precision
if "mlx" in cd.available_backends():
    cd.NeuralGraph(connectome, neuron_model, backend="mlx")
brain = cd.Brain.compose(8, 3, backend="torch", device="cpu")                    # a composed brain
check = cd.conformance(cd.NeuralGraph(connectome, neuron_model, backend="torch"), {0: 1.0})
print(check["backend"], check["max_abs_deviation"])                              # vs the float64 reference
```

## Which hardware

[The default backend](#the-default-backend) holds the rule a new brain follows and the A10G and M4 tables of
`act`, `lesson` and `step`. Small brains spend more time launching GPU operations than doing arithmetic, while
large brains, deep brains and large batches gain from the device. On that A10G, one process with the GPU to
itself, milliseconds per act and per taught lesson of a newborn at each precision, against one CPU core of the
same machine at the CPU path's default:

| Brain | Rows | One CPU core, act and lesson | CUDA float64 | CUDA float32 |
| --- | ---: | ---: | ---: | ---: |
| The chamber's default brain: 251 neurons, 13,120 synapses | 1 | 0.52, 2.68 | 1.61, 7.90 | 1.51, 7.66 |
| The same | 16 | 0.92, 6.94 | 1.99, 9.84 | 1.76, 8.46 |
| The same | 64 | 1.90, 20.6 | 2.58, 18.3 | 2.10, 16.8 |
| The speech size, `Brain.compose(3920, 37, modules=(512,))`: 12,821 neurons, 6.3 million synapses | 1 | 9.39, 405 | 3.78, 29.5 | 3.53, 21.9 |
| The same | 16 | 42.4, 505 | 6.95, 42.3 | 3.92, 23.2 |
| The same | 64 | 123, 815 | 14.1, 70.2 | 5.29, 27.9 |
| The same | 256 | 439, 2,165 | 48.4, 191 | 11.3, 47.2 |
| The same | 1,024 | 2,248, 9,424 | 174, 667 | 36.6, 149 |
| A deep apical brain, `modules=(128, 64, 64, 64), apical=0.25, teaching="deep"`: 535 neurons, 50,048 synapses | 1 | 1.42, 6.32 | 2.31, 10.8 | 1.84, 9.27 |
| The same | 16 | 10.7, 37.4 | 2.75, 12.1 | 2.05, 9.96 |
| The same | 64 | 38.8, 133 | 4.89, 20.9 | 3.49, 16.3 |
| The same | 256 | 152, 524 | 8.55, 33.9 | 3.76, 17.6 |
| The deep anatomy of [`benchmarks/torch_gpu`](../benchmarks/torch_gpu/README.md) under its compartment: 7,956 neurons, 903,520 synapses | 64 | 910, 2,247 | 16.8, 46.6 | 4.26, 15.7 |
| The same | 256 | 3,479, 8,625 | 58.7, 164 | 13.9, 40.3 |

A settle of 32 sweeps from a warm state under a new drive on that device, in milliseconds:

| Brain | Rows | CUDA float64 | CUDA float32 |
| --- | ---: | ---: | ---: |
| The chamber's default brain | 1 | 0.65 | 0.53 |
| The same | 16 | 0.74 | 0.55 |
| The same | 1,024 | 2.43 | 1.03 |
| The speech size | 1 | 1.20 | 0.88 |
| The same | 16 | 3.71 | 1.28 |
| The same | 64 | 8.97 | 2.80 |
| The same | 1,024 | 95.4 | 29.6 |
| The deep apical brain | 16 | 1.50 | 0.85 |
| The same | 256 | 5.76 | 1.31 |
| The deep anatomy | 16 | 6.04 | 2.33 |
| The same | 1,024 | 207 | 48.8 |

A settle is about a third of an act of a brain with the memory, whether its memories run on the host below
[their size](#memories-on-the-device) or on the device from it: 0.55 of 1.76 milliseconds for the chamber's brain
at 16 rows and 1.28 of 3.92 at the speech size. The device's float64 runs at a sixty-fourth of its float32 rate;
check float32 against float64 as below before relying on it. On the small continuing nursery of the
[System 1 CUDA qualification](../benchmarks/system1_cuda/RESULTS.md), CPU with Numba is faster
than CUDA.

On an Apple M4 the rows of one CPU core at its default solve and of MLX, in milliseconds of `act`, `lesson` and
`step`, are in [the default backend](#the-default-backend). A brain without the memory keeps the ordinary solve on the
CPU. In milliseconds per act and per taught lesson of a newborn, MLX against one CPU core with Numba (float32 against
float64, the same sweeps), at a load of about 2.5:

| Brain | Rows | One CPU core, act and lesson | MLX, act and lesson |
| --- | ---: | ---: | ---: |
| A small brain without the memory: 12 inputs, association 32 | 1 | 0.07, 0.26 | 0.66, 2.8 |
| Four regions under an apical compartment, `modules=(128, 64, 64, 64)`, without the memory: 439 neurons, 43,904 synapses | 1 | 0.42, 1.8 | 0.73, 3.3 |
| The same | 16 | 1.9, 5.5 | 0.79, 3.3 |
| The same | 64 | 8.4, 23 | 5.9, 21 |
| The deep anatomy above, with its compartment: 7,956 neurons, 903,520 synapses | 1 | 33, 118 | 99, 260 |
| The same | 16 | 132, 426 | 106, 537 |
| The same | 64 | 248, 650 | 118, 350 |
| The same | 256 | 1,090, 2,680 | 312, 896 |
| The same | 1,024 | 4,550, 11,400 | 1,370, 3,560 |
| The same | 4,096 | 18,000, 49,100 | 7,130, 25,500 |

MLX wins at the speech size at every batch, on four regions under a compartment from about 16 rows, and on the
deep anatomy, which takes the compiled sweep, from about 32 rows. At the chamber's size one synchronization with the
GPU costs about 180 microseconds whatever the work, and a solve of three chunks takes three, so one core stays the
choice there at 16 rows.

So a device costs a small brain time. At one row a new brain of 12 inputs and an association of 32 takes about 4
times one CPU core's time per act and per lesson on MLX, and the chamber's brain about 3 times per act and per lesson on
CUDA; at 16 rows the chamber's brain takes about 1.2 times on MLX, and on CUDA 1.9 times per act and 1.2 per lesson. Keep a
brain on the CPU under the sizes where a device passes one core.

The load moves these readings: above a load of 4 an MLX act
at the speech size has read twice its quiet time.

### The read-speech recipe on a device

The read-speech recipe's brain, `Brain.compose((40, 24), 29, learned_fields=64, modules=(256,),
fields_motor=1.0, learning_surprise=False)`, trains on CUDA float32 and on MLX with 32 and 128
utterances heard side by side as streams, its lessons within the settle's tolerance of the CPU's
and its saves loading on every backend to the same parameters
([on a device](../benchmarks/letters/README.md#on-a-device-issue-314-r3)). Seconds of training
per hour of audio, a lesson and the alignment's greedy act per step:

| Where | 32 streams | 128 streams |
| --- | ---: | ---: |
| NVIDIA A10G, CUDA float32 | 187 | 126 |
| One CPU core of the same machine | 770 | 691 |
| Apple M4, MLX, shared (load 20 to 37) | 456 to 1,083 | 251 to 501 |

Its learned fields read and learn on the host whatever the backend; on the A10G at 128 streams
they take about 60% of a lesson step.

Cap BLAS and Numba threads for parallel workers. To compare two implementations, use identical
dtype, steps and tolerances over several seeds, warm up, alternate the order, synchronize
around each timed call, and compare residuals and states too.

## Many lives at once

Search and self-play settle many positions or games each second. Which way to run them depends on whether
they are one brain's streams, sharing what they learn, or brains of their own.

| You want | Use |
| --- | --- |
| One player in many games, a search over positions: one brain, many streams | `Brain.compose`, one row per stream in `step` or `act`, on a device |
| Many brains of their own on a CUDA card: a population, a league, independent seeds | `Lives(inputs, actions, lives, backend="torch")` |
| Many brains of their own on a Mac or a CPU machine | a process pool, one thread per process, or `Lives(..., backend="cpu")` in one process |

```python
import numpy as np

lives = cd.Lives(35, 18, 8, backend="torch")            # 8 brains of seeds 0 to 7, on CUDA where there is one
observations = np.random.default_rng(0).random((8, 35))
actions = lives.step(observations)                      # one row per life in, one action per life out
actions = lives.step(observations, reward=np.ones(8))   # each life learns from its own outcome
actions = lives.step(observations, reward=np.ones(8), teacher=np.arange(8))  # and from its own label
lone = lives.life(5)                                    # life 5 as a lone Brain on the CPU, to save

living = cd.Lives(35, 18, 8, backend="torch", arousal=True)  # continuing lives, each with its own mood
actions = living.live(observations)                     # the first moment: no outcome yet
done = np.array([False, True, False, False, False, False, False, False])
actions = living.live(observations, reward=np.ones(8), done=done)  # life 1's episode ends
```

`Lives` composes each life as `Brain.compose(inputs, actions, seed=s, **options)`: the default composition, its
store of place records among it (`episodic=False` composes without the store, as `Lives` did in 0.96). Each life has its own synapses and biases, its own optimizer (momentum, second
moments and their counts), critic, eligibility, recall memory (its slots, its address and position maps), carried
gains, working trace and random generator. On a torch device the lives are the streams of one brain whose
parameters carry a lives axis:

- a settle reads every stream through its own life's synapses and holds each life where its own stream meets the
  tolerance, and a rescue on the damping ladder runs life by life;
- a lesson judges each life by its own answer under `learning_surprise`, runs its phases on every life and
  updates the lives it taught, each at its own optimizer's count;
- the recall memory's address and position lessons, the carried gains' steps and a reward's update move each life
  on its own row and its own dopamine;
- with the store (`episodic=True`) each life's place records rest on the device with a lives axis: an outcome
  writes the records of the lives it teaches and decays the residual of those whose outcome it is, the outcome that
  wakes a calm life writes that life's records and settles it again under them while the others hold, with
  `memory_fade` each life's records fade by its own depths and recent use at each of its moments, and with
  `maturity` each life's use scales its own actor steps and engages its own surprise gate on its records;
- in `live` each life keeps its own `Arousal` on the host: the lives whose action was sampled learn from their
  outcomes while the others hold their state and parameters, the calm lives' eligibility fades by one moment and
  their ended streams start from rest, one qualified settle answers every life (a calm life's forecast is its
  answer), and each life answers greedily or samples at the temperatures its own arousal sets, the heated slot and
  the sample drawn from its own generator, its eligibility phases pushed at its own temperatures.

Each life follows the lone brain of its seed on that device ([`tests/test_lives.py`](../tests/test_lives.py)). In
float64 every action is the same and every learned part (synapses, biases, critic, optimizer, recall maps, carried
gains) agrees within 1e-9 over 30 moments of rewards, endings and teaching, and over 160 moments of the chamber's
composition within 1e-8; on an NVIDIA L4 over 48 moments within 1e-10. In float32 every action is the same over
those moments and the parts agree within the spread float32 gives a lone brain's own two memory paths (its
memories on the host against on the device: 1e-2 in the recall maps over 48 moments, where a float32 life on the
L4 reads 4e-3). On the CPU each life is the lone brain itself, to the bit. Continuing lives (`arousal=True`) that
`live` 30 moments of sparse rewards and endings, with a teacher's step at every fifth, keep every action, every
mood and heated slot of their lone brains, and every learned part and every arousal reading within 1e-9, with one
or two motor slots, with the recall memory and without; a life that `live` leaves calm takes no outcome in a
`step` or `learn`, as its lone brain takes none. With the store, alone, fading, under maturity and with both, the
same holds over 30 moments of `live` in which an occasional outcome wakes a calm life and writes its records, and
over 24 moments of `step`: every action and mood its lone brain's, every learned part, record, noise scale, depth
and use within 1e-9.

A life leaves with `lives.life(i, stream=True)`: a lone brain that carries its stream as it stands, its pending
sampled action and that action's eligibility among it, and continues the life where it left. A life joins with
`lives.join(i, brain)`, taking the place of life `i` with its stream afresh, so a life whose episode finished makes
room for another without holding the batch. Lives that left mid-stream, sampled and calm, continue on their own as
their lone brains over 12 moments of `live`, every action the same and every learned part within 1e-9; a brain
that joins after living alone and a `reset` lives on in the batch as alone over 14 moments, every action and mood
the same and every learned part within 1e-9.

Seconds per life and moment of that composition stepping its own observations and rewards, read with
[`benchmarks/device/lives.py`](../benchmarks/device/lives.py) on one AWS g6.xlarge (an NVIDIA L4 and four vCPUs
of an AMD EPYC 7R13) and on an Apple M4 at a load of 4:

| Way | Machine | 12 lives | 128 lives | 512 | 1,024 |
| --- | --- | ---: | ---: | ---: | ---: |
| `Lives`, `backend="torch"` | L4 | 1.18 ms | 0.118 ms | 0.043 ms | 0.049 ms |
| One brain's streams (shared parameters) | L4 | 0.619 ms | 0.119 ms | 0.045 ms | 0.034 ms |
| Process pool, four CPU processes | L4 box | 0.874 ms | 0.876 ms |  |  |
| Each brain in turn, one core | L4 box | 2.35 ms | 2.37 ms |  |  |
| Each brain in turn on the L4 | L4 | 6.52 ms | 6.66 ms |  |  |
| One brain's streams on MLX (shared parameters) | M4 | 0.690 ms | 0.156 ms |  |  |
| Process pool, two CPU processes | M4 | 0.595 ms | 0.662 ms |  |  |
| Each brain in turn, one core | M4 | 1.42 ms | 0.979 ms |  |  |
| `Lives` on the torch CPU device | M4 | 1.17 ms | 1.21 ms |  |  |

The same with a teacher's label at every fourth moment (`--teach 4`):

| Way | Machine | 12 lives | 128 lives | 512 | 1,024 |
| --- | --- | ---: | ---: | ---: | ---: |
| `Lives`, `backend="torch"` | L4 | 1.45 ms | 0.147 ms | 0.056 ms | 0.067 ms |
| One brain's streams (shared parameters) | L4 | 0.789 ms | 0.153 ms | 0.056 ms | 0.040 ms |
| Process pool, four CPU processes | L4 box | 1.12 ms | 1.13 ms |  |  |
| Each brain in turn, one core | L4 box | 2.97 ms | 3.03 ms |  |  |
| Each brain in turn on the L4 | L4 | 7.86 ms | 7.98 ms |  |  |
| One brain's streams on MLX (shared parameters) | M4 | 0.572 ms | 0.128 ms |  |  |
| Process pool, two CPU processes | M4 | 0.808 ms | 0.727 ms |  |  |
| Each brain in turn, one core | M4 | 1.56 ms | 1.48 ms |  |  |
| `Lives` on the torch CPU device | M4 | 1.48 ms | 1.32 ms |  |  |

Continuing lives that `live` (`--live`: `arousal=True` with a youth of 10 moments, each life's own rewards and
episode ends, one moment in 20 ending one), over 20 moments after 12, past the youth, where 52 to 54 in 100 of
the lives' moments are calm:

| Way | Machine | 12 lives | 128 lives | 1,024 |
| --- | --- | ---: | ---: | ---: |
| `Lives`, `backend="torch"` | L4 | 1.31 ms | 0.163 ms | 0.090 ms |
| Process pool, four CPU processes | L4 box | 0.422 ms | 0.390 ms | 0.408 ms |
| Each brain in turn, one core | L4 box | 1.03 ms | 1.03 ms |  |

With the store (`--composition default`, 630 place records a life), and with the store fading under maturity
(`--fade 0.01 --maturity`), the same `live` reads:

| Way | Machine | 12 lives | 128 lives | 1,024 |
| --- | --- | ---: | ---: | ---: |
| `Lives`, `backend="torch"`, the store | L4 | 1.73 ms | 0.220 ms | 0.107 ms |
| Process pool, four CPU processes, the store | L4 box | 0.497 ms | 0.461 ms | 0.460 ms |
| `Lives`, `backend="torch"`, fading under maturity | L4 | 2.09 ms | 0.239 ms | 0.122 ms |
| Process pool, four CPU processes, fading under maturity | L4 box | 0.529 ms | 0.535 ms | 0.509 ms |

A lone brain's calm moment skips the eligibility phases and the update, so `live` in the pool costs about half a
`step`. A batch with any aroused life runs the phases for every life and holds the calm ones, so `Lives` pays
for its aroused lives across the batch: 2.4 times as fast as the pool at 128 lives and 4.5 times at 1,024 (2.1
and 4.3 times with the store, 2.2 and 4.2 times fading under maturity), and the pool is faster at 12.

On the L4, `Lives` runs at the speed of one brain's streams at 128 and 512 brains and at 1.4 times their time at
1,024, where every life updates synapses of its own and the streams share one update. Against the four-process pool
it is 7.4 times as fast at 128 brains and 17.8 times at 1,024 (7.7 and 17.0 times with the teacher). At 12 lives
the pool is faster. A lesson of lives runs its phases on every life, where one brain's streams run them on the rows
it teaches.

MLX has no lives axis in the library. What one would cost, read on the Apple M4 at a load of 13 to 40 (other work
on the machine; the CPU readings carry it) with
[`benchmarks/device/lives_mlx_cost.py`](../benchmarks/device/lives_mlx_cost.py) and `benchmarks/device/lives.py`,
seconds per life and moment of the composition without the store:

| Way on the M4 | 12 lives | 128 lives | 1,024 |
| --- | ---: | ---: | ---: |
| One brain's streams on MLX (shared parameters), `step`: the bound | 0.921 ms | 0.232 ms | 0.170 ms |
| One sweep's transport on MLX, each life's own blocks against one brain's | 1.26 times | 1.45 times | 3.34 times |
| Process pool, two CPU processes, `step` | 1.63 ms | 1.89 ms | 1.68 ms |
| Process pool, two CPU processes, `live` | 2.02 ms | 1.44 ms | 1.75 ms |
| `Lives` on the torch CPU device, `step` | 4.01 ms | 3.39 ms | 3.45 ms |

A lives axis on MLX would read between the bound and the bound times the transport's factor, at most 0.34 ms at
128 lives and 0.57 ms at 1,024: three to ten times as fast as the M4's pool here, and two to twelve times the time
of `Lives` on one L4. It would take MLX versions of the per-life transport and stop, the per-life optimizer, critic, recall maps,
carried gains, store and nudges, and somewhere for the settles' float64 checks: the Apple GPU has no float64, so
they would run on the host or MLX's CPU stream.

`Lives` holds the default composition, its store and its heard history among it, and the same without the store
(`episodic=False`) or the history (`history_decays=()`), with or without the recall memory, the carried gains, arousal,
`memory_fade` and maturity, at any teaching point, with one or several regions, observers and motor slots. It refuses,
naming the part: `memory_choice` and a store with a separator; a learned or fixed code and image inputs; an apical
compartment; rehearsal and qualified teaching; the efference copy; a held message (`message`) and long-term
records (`longterm`); tied projections (`tied`); `actor_entropy`; a recall memory with several heads, a
window, the command, the read of when, tokens or the cortex at unit norm; MPS and MLX. A moment any life
refuses is refused for every life. Each life holds its own synapses, their dense blocks, two optimizer moments and an eligibility, five arrays
of one float per synapse, and in float32 a float64 copy of the blocks for the settles' checks.

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
hundredth of the tolerance, and by the reference below that. So does a solve on a float32 torch state,
by that state's check in float32 (`own_precision=True`); a finite lesson's diagnostics and the gain
its nudged phases hold are read the same way, so a float32 brain learns without a float64 copy of its
kernel. A positive movement
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
launch, a softmax nudge over equal groups side by side another, and the apical normalization three:
the sums of squares over tiles of a pool's members, each pool's gain, the drive. `CADENCE_TRITON=0`
keeps cuBLAS products and one torch operation at a time. The sweeps of a settle and a residual check
are captured once as CUDA graphs and replayed (`CADENCE_CUDA_GRAPHS=0` runs them eagerly; a replay is
bitwise the eager run), drives and masks reach the device through pinned host buffers, and a sweep
allocates nothing. A settle of no sweeps from a resident state published under no mask returns that
state.

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
