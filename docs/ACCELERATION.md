# GPU execution and parallel experience

Cadence can accelerate repair with CPU or GPU tensors and run independent
simulated lives in separate processes. The brain's interface stays the same:
`settle` queries, `step` retains qualified activity, and `observe` admits a
target through joint state/parameter repair, labeled as a witness or an estimate.
`observe_batch` admits a group of labeled examples with shared parameters and
private experience states. `bootstrap` replays witnesses, optionally in batches,
and checks unclamped predictions.

More width or recursive depth increases capacity and cost. Whether it improves
a task must be tested; adding layers does not guarantee successful learning.
Acceleration changes execution, not the information supplied, the learning
objective or the need for a useful teaching stream.

Choose the layout before assuming a faster device will solve the latency
problem. Input-only flat queries can require very little repair; ordinary
composition and recursive observation introduce different coupling costs.
See [depth, latency and useful work](PERFORMANCE.md) for versioned demo evidence,
a bounded query comparison and the distinction between answering and learning.

## Select execution explicitly

The default `Cortex()` uses Python float64 and has no runtime dependencies.
Install the optional tensor backend for the same release when needed:

```sh
python -m pip install "cadence-net[gpu]==0.60.0"
```

For repository scripts, first obtain the
[tagged source checkout](../examples/README.md#get-the-example-sources).

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
from cadence import Brain, Cortex, bootstrap

layout = Cortex(seed=2, device="python")
signal = layout.input("signal", shape=1)
base = layout.column("perception", patches=4, inputs=signal)
observer = layout.observer("reflection", patches=2, inputs=signal, observes=base)
layout.output("answer", shape=1, reads=observer)
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
`Reinforcement` supplies Q targets, with one-step credit by default, and uses
the same batch execution with `source="estimate"`; its next-action queries
also count as work.

The [batch bootstrap example](../examples/batch_bootstrap.py) measures a small
supervised relation with 32 teaching rows, eight readiness checks and 16 fresh
test rows. Its 12 processing patches and four observing patches receive eight
numeric sensors and expose two outputs. From the tagged source checkout root:

```sh
python -m pip install -e ".[gpu]"
python examples/batch_bootstrap.py --devices python cpu mps --batch-size 8 --repeats 3 --threads 1
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
python examples/parallel_bootstrap.py --lives 4 --workers 2
```

The [complete example](../examples/parallel_bootstrap.py) creates independent
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

## Measured starting points

### NVIDIA hardware qualification

On 2026-10-01, an NVIDIA RTX 4000 Ada Generation Laptop GPU with 12,282 MiB
of VRAM passed all 67 selected CUDA cases in `test_tensor_math.py` and
`test_batch_tensor.py`, with no skips. The Windows run used Python 3.13.2,
driver 595.95, PyTorch 2.11.0+cu128 (CUDA runtime 12.8), and both float64 and
float32 proposals. The final [audited receipt](../examples/receipts/cuda_audit_verified.json)
records a full run with **937 passed and 34 MPS skips**, including all 67 CUDA
cases, on commit `8892927`. It identifies the tested Git tree, hardware and
library versions, individual CUDA outcomes, and unchanged source hashes before
and after testing. MPS and additional CUDA device indices were unavailable on
this single-GPU laptop.

Runtime is also part of the evidence. In that final full run, the pytest
invocation took **297.72 seconds**, including collection but excluding the
collector's imports and CUDA preflight. Its JUnit test-case durations include
the complete fixture, setup/teardown and assertions:

| Acquisition fixture | CUDA float64 | CUDA float32 |
| --- | ---: | ---: |
| Single-example bootstrap, held-out recall and live learning (6 patches) | 3.479 s | 124.874 s |
| Batch bootstrap, recall, checkpoint transfer and live learning (3 patches) | 1.961 s | 2.045 s |

These are `test_bootstrap_held_out_recall_and_live_learning_on_device` in
`test_tensor_math.py` and
`test_public_batch_bootstrap_recall_checkpoint_transfer_and_live_learning` in
`test_batch_tensor.py`. Both precisions passed, but float32 took about **36
times as long** in the single-example fixture. The receipt does not retain
per-solve work or phase timings, so it does not establish the cause or equal
work across precision trajectories.
The batch fixture uses a different graph and task; comparing the two rows
does not measure a batching speedup.

The four retained runs also show substantial variation: this single-example
fixture ranged from **3.479 to 16.280 seconds** in float64 and **124.874 to
347.800 seconds** in float32. Those runs used different suite selections or
revisions and uncontrolled timing conditions. They are observations, not a
latency distribution. A performance follow-up needs repeated matched Python,
CPU tensor and CUDA workloads, separate initialization and complete-call
timings, and per-solve work/refinement counts alongside qualification and
unclamped prediction error.

The checks cover independent scalar finite-difference derivatives, recursive
feedback, original frozen parameters and witness clamps, strict reference
qualification, overflow/refusal, checkpoint transfer, batch admission/retry
custody, and subsequent unclamped recall and live learning. The CPU/MPS
checks also select CUDA when available. Run the current CUDA subset from
a clean, committed source checkout installed with
`python -m pip install -e ".[dev]"` and a CUDA-enabled PyTorch build:

```sh
python -X utf8 examples/cuda_qualification.py --out data/cuda-qualification.json
```

The [collector](../examples/cuda_qualification.py) requires actual CUDA and
both precision cases, records the tested Git commit and source hashes before
and after testing, and fails on skipped CUDA cases or changed sources. Use
`--full` for the complete test suite. Choose a new output path for each run;
existing receipts are never overwritten. Refusal tests require a tensor sweep
before checking rollback; derivative checks also assert actual tensor dtypes.

This is correctness and small-fixture acquisition evidence for
[issue #64](https://github.com/muellerberndt/cadence/issues/64), not a speedup or
deployment qualification. Peak host/device memory, temporary and cached-index
growth, real allocation-failure recovery, full perception-to-action latency
tails, and matched-workload performance comparisons remain unmeasured here.
Test durations are not production p50/p95/p99 latency measurements.

The [original receipt](../examples/receipts/cuda_qualification.json) retains
one reference-engine failure from the first full run. The deep learning
fixture in `test_learning_basics.py` qualified after 476 sweeps on this
Windows machine, while the same sources need 777 sweeps on an Apple M4 with
Python 3.13, so a 512-sweep cap refused the example on one platform and
admitted it on the other. Sweep counts near a cap are platform-dependent; the
rollback test therefore uses a one-sweep cap and keeps its default-budget
retry. The repair rule and admission tolerance are the same on both platforms.

An [intermediate collector failure](../examples/receipts/cuda_audit_encoding_failure.json)
is also retained: Python's `-X utf8` setting did not reach CLI subprocesses on
Windows. The collector now propagates that setting, and the final full run
above passed with the correction.

### Thread, device and worker measurements

One three-batch `observe_batch` comparison took 934.0 seconds with 64 Torch
threads and 1,118.9 seconds with 128, with identical accepted sweep counts.
This establishes about 1.20× for those two settings on that workload, not an
optimal thread count or a physical-core count. Benchmark thread and worker
counts on complete calls, including reference qualification, before scaling.

The measurements below use individual witness admissions. They are not batch
benchmarks and do not predict a speedup from changing `batch_size`.

On an Apple M4 with Python 3.13 and PyTorch 2.14, a bounded comparison used
64 inputs, four output coordinates, three seeds, two repetitions and tolerance
`1e-6`. Each repeat queried twice, admitted two supplied witnesses, then queried
twice more. All 432 public calls qualified; final checks and refinement are
included in the timings below.

| Layout | CPU tensor / Python speedup | MPS / Python speedup |
| --- | ---: | ---: |
| 8 patches | 2.47× | 0.25× |
| 64 patches | 9.16× | 1.12× |
| 256 patches | 19.55× | 2.31× |
| 128 patches + 32 observers | 15.34× | 2.74× |

These are medians of paired **witness-update** time ratios, not complete skill
acquisition or query-only speedups. A ratio below one is a slowdown. Over the
complete six-call sequence, the 256-patch ratios were 16.25× for CPU tensors and
2.17× for MPS; the recursive layout ratios were 13.63× and 2.40×. Initialization
was measured separately. Devices did not produce bit-identical trajectories;
maximum paired output difference after the witness updates was `8.20e-7`.

For these sizes, try `device="cpu"` first. GPU support is useful without being
the fastest option for every layout. These measurements establish execution
improvements on the stated workload, not an architectural advantage from depth.

The independent-life example also ran 16 lives with one, two and four workers,
three repetitions each. Median complete process times, including startup and
JSON output, were 0.795, 0.496 and 0.340 seconds: 1.60× and 2.34× speedups.
All 144 life outcomes matched serial execution apart from timing and process
IDs. Small workloads can still lose to process overhead on other machines.
