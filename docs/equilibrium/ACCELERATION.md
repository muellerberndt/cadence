# GPU execution and parallel experience

Cadence can accelerate repair with CPU or GPU tensors and run independent
simulated lives in separate processes. The brain's interface stays the same:
`settle` queries, `step` retains qualified activity, and `observe` admits a
target through joint state/parameter repair, labeled as a witness or an estimate.
`observe_batch` admits a group of labeled examples with shared parameters and
private experience states. `bootstrap` replays witnesses, optionally in batches,
and checks unclamped predictions.

Choose a connected graph and a useful teaching stream before selecting an
accelerator. Population sizes and wiring determine capacity and cost.
Acceleration changes execution while preserving the same energy, learning rule
and qualification. See [capability and cost](PERFORMANCE.md) for query/learning
measurements and versioned demo evidence.

## Select execution explicitly

The experimental `Cortex()` defaults to a standard-library float64 solver.
The containing Cadence distribution also requires NumPy.
Install the optional tensor backend for the same release when needed:

```sh
python -m pip install "cadence-net[accel]==0.80.0"
```

Use the [quickstart](QUICKSTART.md) for the current `0.80.0` installation.
The device measurements below retain their recorded source versions; they are
not current-version or all-workload performance guarantees.

| `device` | Proposal arithmetic | Default `dtype` |
| --- | --- | --- |
| `"python"` | Standard-library reference engine | `"float64"` only |
| `"cpu"` | PyTorch tensors on CPU | `"float64"`; also supports `"float32"` |
| `"mps"` | PyTorch tensors on an available Apple GPU | `"float32"` only |
| `"cuda"` or `"cuda:N"` | PyTorch tensors on an available NVIDIA GPU | `"float64"`; also supports `"float32"` |

`"cuda"` resolves to `"cuda:0"`. Set `dtype` explicitly to override the selected
device's default where supported. PyTorch supplies tensor arithmetic; Cadence
computes its own analytic derivatives and repair steps. There is no PyTorch
neural-network module, autograd optimizer or second learning algorithm here.

This small reference example is runnable with the base installation:

```python
from cadence.experimental.equilibrium import Brain, Cortex, bootstrap

layout = Cortex(seed=2, device="python")
signal = layout.input("signal", shape=1)
base = layout.column("perception", patches=4, inputs=signal)
response = layout.column("response", patches=2, inputs=(signal, base))
layout.output("answer", shape=1, reads=response)
brain = layout.build()

examples = [({"signal": [x]}, {"answer": [x]}) for x in (-0.8, 0.8)]
checks = [({"signal": [x]}, {"answer": [x]}) for x in (-0.4, 0.4)]
report = bootstrap(brain, examples, checks=checks, max_error=0.2)
assert report["passed"], report
saved = brain.snapshot()
```

Set `device="mps"` or `device="cuda"` on the same constructor to bootstrap on
the corresponding GPU. Use `device="cpu"` to test tensor execution without a
GPU. The hardware and tensor library must support the selected configuration.
Construction and checkpoint loading do not import PyTorch or reserve a device;
the first solve does. Missing PyTorch raises `ImportError`, and unavailable
hardware raises `ValueError`. Cadence does not silently choose another device.

The device comparison below exercised CPU float64 and Apple MPS float32.
Application campaigns also report CUDA float64 repair measurements; retain
their exact source, hardware and workload when interpreting them. Neither
those measurements nor the table below qualifies every deployment workload.

## Batch experience on one device

`brain.observe_batch(examples)` and `bootstrap(..., batch_size=...)` use one
brain's shared parameters with a private state for each example. CPU, MPS and
CUDA tensor execution vectorizes the row dimension as well as eligible patch
arithmetic. Observed and observing populations within every row still settle
together; the shared parameters repair against the mean row objective with
one fixed pre-batch anchor.

A complete batch must qualify against the original float64 objective. Each
row's state residual is tested **without** the averaging factor, so batch size
never weakens its tolerance. A successful batch commits shared parameters and
one event identity, preserving the pre-call live state. The same operation
runs on the Python reference engine without optional dependencies.

Start with a bounded size such as 4 or 8, then measure your actual workload.
These are trial sizes, not universally optimal defaults. Temporary state and
activation memory grows with the number of rows. Group size also changes the
learning trajectory compared with ordered single-example admissions, so measure
held-out accuracy and retention along with throughput. Count example
presentations, atomic `updates`, refused solves and all checking/refinement work.

This is one-device batching, not distributed multi-GPU learning, averaging
checkpoints or concurrent calls to one brain. Continue to serialize admissions.
Independent lives can still use separate processes. Neither batching nor
parallel simulation supplies a missing memory or temporal-credit mechanism.
`Reinforcement` explicitly supplies one-step Q targets and uses the same batch
execution with `source="estimate"`; its next-action queries also count as work.

The [batch bootstrap example](../../examples/equilibrium/batch_bootstrap.py) measures a small
supervised relation with 32 teaching rows, eight readiness checks and 16 fresh
test rows. Its two ordinary populations contain 12 and four patches, receive
eight numeric sensors and expose two outputs. From the repository root:

```sh
python -m pip install -e ".[accel]"
python examples/equilibrium/batch_bootstrap.py --devices python cpu mps --batch-size 8 --repeats 3 --threads 1
```

List only devices available on your machine; unavailable devices fail explicitly.
Omit `--devices` for a Python-only run without tensor dependencies. The JSON
separates construction/first-solve warmup, full bootstrapping (including every
readiness check), and unclamped fresh-test times. Process startup and imports
are outside those timers. Compare their sum for complete case work and inspect
`passed`, errors, presentations, updates and repair work alongside speed.
The fixed fixtures and readiness criterion are shared across devices; this is
neither a game benchmark nor evidence of a depth advantage. Changing batch size
changes the learning path and needs its own acquisition check.

## Keep admission precise

All tensor proposals are finally checked by the Python float64 reference engine
against the **original** inputs, exact clamps, original parameter anchors and,
for queries, original frozen weights/biases. Float32 proposal arithmetic never
loosens the configured stationarity tolerance. Reference refinement can use
the remaining accepted-sweep allowance. If the candidate raises the original
float64 energy beyond the permitted rounding allowance, reference repair
restarts from the original coordinates within that remaining allowance.
For float32, the device phase accepts at most `max(1, budget // 2)` sweeps
when the budget is positive, reserving at least half of any budget of two or
more for reference refinement. Float64 may use the full device allowance.
The device stopping hint is `max(tolerance, 64 * dtype_epsilon)`; it does not
change the final admission tolerance. The complete solve still obeys `budget`;
a refused proposal is never committed.

The two precisions need not take identical steps or reach the same stationary
point. Inputs and parameters must fit the selected proposal dtype. Float32
may need more reference refinement at a tight tolerance, and unsuitable
numeric magnitudes can raise `ValueError` without changing continuation.

For tensor solves, `result["execution"]` records `device`, `dtype`, `torch`,
`tensor_sweeps`, `reference_sweeps`, `reference_evaluations` and
`reference_restart`. The ordinary `work` and `sweeps` totals include device
and reference work. `energy`, `errors`, `stationarity` and `qualified` are the
final reference diagnostics. `energy_history` combines the approximate device
trajectory and any reference refinement; it is not a float64 certificate of
monotonic decrease at every intermediate device step.

## Move a continuation between devices

Use an explicit override when loading a compatible checkpoint:

```python
reference = Brain.from_snapshot(saved, device="python")
assert reference.state == brain.state
assert reference.weights == brain.weights
assert reference.biases == brain.biases
assert reference.inspect()["admissions"] == brain.inspect()["admissions"]
```

For example, move the same saved brain to an Apple GPU:

<!-- not-run: Requires the optional PyTorch installation and available MPS hardware. -->
```python
gpu = Brain.from_snapshot(saved, device="mps")
result = gpu.settle({"signal": [0.4]})
assert result["qualified"], result
print(result["outputs"], result["execution"])
```

The entire original checkpoint is validated before applying execution overrides.
This retains its arrays and admission cursor exactly, then changes configuration
and fingerprint. A device override without `dtype` selects that device's
default; a dtype-only override keeps the saved device. With both omitted,
saved settings are retained. `restore` still requires the exact complete
configuration. Source compatibility remains strict: transfer between devices
is not permission to load a checkpoint from different engine sources.

A snapshot preserves arrays and implementation identity exactly, but tensor
arithmetic can vary with hardware and PyTorch version. In particular, PyTorch
documents that [CUDA `index_add_` can be nondeterministic](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.index_add_.html).
A seed alone therefore does not guarantee bitwise GPU replay. Record hardware,
library versions and execution settings alongside experiment receipts. Cadence
does not change global deterministic flags; the default Python engine does
not use these tensor reductions.

Execution overrides above apply to a `Brain` checkpoint. A
`Reinforcement` checkpoint also owns replay, RNG and pending-action state, and
its loader has no device/dtype override. Choose the device when building its
brain; do not replace a whole-life restore with a brain-only transfer.

## Parallel simulated lives

Use the spawn context for workers that use torch. A forked worker can inherit
initialized OpenMP state and deadlock before its first batch. Bound each
worker's tensor/BLAS threads as well as the process count; otherwise independent
lives can oversubscribe the same CPU or accelerator.

From the repository with Cadence installed, run:

```sh
python examples/equilibrium/parallel_bootstrap.py --lives 4 --workers 2
```

The [complete example](../../examples/equilibrium/parallel_bootstrap.py) creates independent
point-body simulators and brains in spawned CPU processes. Each brain
bootstraps a small relation between position, velocity and measured next
position; live queries predict fresh outcomes before those outcomes become
new witnesses. Its JSON report includes readiness, counted work, admissions,
fresh predictions, wall time and checkpoint digests. This is a dynamics lesson,
not a learned walking policy or proof of a depth advantage.

Each worker owns its brain. Experiences within a brain stay ordered because
one witness changes the anchors used by the next. Sharing a mutable brain
between workers or averaging checkpoints does not preserve this rule.
Independent environment collectors can instead send an ordered witness stream
to one owner; the application must preserve event identity and provenance.
Do not start one GPU process per CPU core: many small jobs can contend for the
same accelerator. This example deliberately uses the dependency-free engine.

## Responsive live operation

`LiveController` keeps one callback on a serial worker while the application
submits sensory observations and reads fresh qualified commands. Only one
observation can wait behind the active callback; a newer pending observation
replaces it. Keep reward and transition records in a separate lossless queue.
The worker must exclusively own its brain and any reinforcement helper.

Command age starts at sensory submission, so queue and solve time consume
`max_age`. Expired, refused or malformed completions use the declared fallback.
No call cancels an active solve, and a thread does not remove Python's GIL or
provide hard real-time deadlines. Submission copying/validation and lock
acquisition also cost time. Measure callback latency, pending replacements,
command age and fallback duration along with rendering rate. `slew` can limit
actuator changes toward a selected target; it supplies no neural decisions.
See [live-system setup](LIVE.md).

## Measure the complete improvement

Compare the same starting arrays, inputs, witness sequence, stopping criteria
and task checks. Warm up each device separately and disclose startup cost.
Measure wall time around complete calls, including tensor transfers, device
synchronization and final reference checks. Record device/dtype, hardware,
library versions, qualification/refusal counts, sweep/work totals, acquisition
and retention alongside speed. Count repeated bootstrap presentations
separately from new simulator experience.

Small graphs may run faster in Python because dispatch and synchronization
cost more than their arithmetic. Wider graphs can provide more parallel work;
additional error-observation levels also add dependencies. Benchmark the
actual intended layout before choosing hardware or increasing worker count.
Changing `tolerance` changes what gets admitted and must not be hidden inside
a speed comparison.

## Measure before choosing hardware

Run the CUDA subset from a clean, committed development checkout installed with
`python -m pip install -e ".[dev]"` and a CUDA-enabled PyTorch build:

```sh
python -X utf8 examples/equilibrium/cuda_qualification.py --out data/cuda-qualification.json
```

The [collector](../../examples/equilibrium/cuda_qualification.py) requires actual
CUDA and both precision cases, records the tested Git commit and source hashes
before and after testing, and fails on skipped CUDA cases or changed sources.
Use `--full` for the complete test suite. Choose a new output path for each run;
existing receipts are never overwritten. The checks cover independent scalar
finite-difference derivatives, recursive feedback, frozen parameters and witness
clamps, strict reference qualification, overflow and refusal, checkpoint
transfer, batch admission and retry custody, and subsequent unclamped recall and
live learning. Written receipts accumulate under
[examples/equilibrium/receipts](../../examples/equilibrium/receipts), each
identifying its tree, hardware and library versions.

What the recorded runs establish is correctness and small-fixture acquisition,
not a speedup or a deployment qualification. Peak host and device memory,
allocation-failure recovery, perception-to-action latency tails and
matched-workload comparisons are unmeasured, and test durations are not
production latency percentiles.

Four observations are worth carrying into your own measurement:

- **Precision costs more than it looks.** float32 can take far longer than
  float64 on the same fixture, because the trajectories differ; the receipts do
  not retain per-solve work, so they do not establish the cause.
- **Repeat before believing a timing.** The same fixture varied by a factor of
  several across retained runs under uncontrolled conditions.
- **Sweep counts near a cap are platform-dependent.** The same sources can
  qualify in a few hundred sweeps on one machine and need half again as many on
  another, so a cap that admits an example on one platform refuses it on the
  other. The repair rule and admission tolerance are identical; only the count
  moves.
- **Small graphs lose to overhead.** At a few patches, a GPU is slower than
  Python; the CPU tensor backend wins across a wide range of layouts, and the
  margin grows with width. Try `device="cpu"` first, and benchmark thread,
  worker and device counts on complete calls, including reference
  qualification, before scaling. Process workers also pay startup, so a small
  parallel workload can lose to serial execution.

Measure with the layout you intend to run, and keep `tolerance` fixed across a
speed comparison: changing it changes what gets admitted.
