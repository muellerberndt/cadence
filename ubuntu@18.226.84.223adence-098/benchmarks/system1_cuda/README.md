# System 1 CUDA qualification

This bounded instrument compares the same continuing `Brain.compose` nursery on
CPU with Numba, CUDA float64, and CUDA float32. It is separate from the population
solver measurements in issue #64. It measures neither Doom nor speech. The
sequence integration in #121 is pending. Application access is now available;
the supplementary keyword compatibility checks are recorded in `RESULTS.md`.
The Fable sequence campaign still requires the measured integration and the
training hold's dependency checks in #121; keyword throughput cannot replace it.

The [RTX 4000 Ada results](RESULTS.md) include the exact source revisions,
before/after runtimes, latency tails, memory, quality and failed development work.
The linked `.json.gz` record is ordinary gzip-compressed JSON, readable with
Python's standard `gzip` and `json` modules.

## Frozen protocol

`protocol.json` declares seeds, information, topology, learning configuration,
quality gates and work limits before the campaign. One acquired brain survives
stable use, reduced input amplitude, scheduled corrections and continued use.
Actual rewards describe the preceding executed action; teacher labels describe
the current observation. Every live call uses working and associative memory.
The teacher schedule is declared orchestration, not automatic failure detection.
All stages continue actual-reward learning. Held-out acquisition and retention
queries are separate, memory-free graph controls, not the whole brain lifecycle.

The noise and two classes define a small synthetic nursery, not an acquired
perceptual representation. Seeds vary the brain and continuing stream; training
and held-out data seeds stay matched across revisions and backends. Finite
supervised phases use their existing contract; free answers must satisfy the
unchanged 0.003 residual tolerance. Strict qualified learning has separate
independent derivative and refusal tests in `tests/test_qualified_learning.py`
and `tests/test_system1_cuda.py`.

`bottleneck.py` measures a larger coupled graph with returning observers. Its
finite-phase cap warnings remain in the logs; it establishes a bottleneck, not
successful acquisition. `transport.py` isolates fixed 32-sweep work across two
topologies and three batch sizes. It retains potentials and original-equation
residuals without claiming those finite states qualify as answers. Those controls
separate kernel improvement from changes in adaptive stopping or learning.

## Reproduce

Use the repository's development environment plus a PyTorch build that supports
the actual CUDA hardware. Both source checkouts must be clean and committed.
Run one campaign at a time; the parent alternates baseline/candidate nursery order
and starts a fresh process for every source/backend/seed. No paid compute is used.

```sh
git worktree add --detach .venv/issue98-baseline 5cd3b3d
python benchmarks/system1_cuda/campaign.py --baseline .venv/issue98-baseline --candidate . --out .venv/system1-cuda-run
python benchmarks/system1_cuda/verify.py .venv/system1-cuda-run --out .venv/system1-cuda-result.json
```

The campaign refuses an existing output directory. Every worker records the
commit, LF-normalized SHA-256 of all implementation files, protocol and collector
hashes, and checks sources again after execution. All failures and process wall
times remain in the output; a failed job does not suppress later jobs. The
verifier independently recomputes quality gates, latency percentiles, throughput,
hash bindings and fixed-work baseline/candidate/reference state deviations. It
also checks the ordered job census, device/dtype, checkpoint-bound effective
settings, per-row admission diagnostics, live scores and paired actions/work.
These artifact-consistency checks do not authenticate witnessed experience.

CUDA float32 uses float32 settling and the existing float64 parameter/optimizer
contract. It checks the original float64 equations for qualification. TF32 is
disabled. Device and dtype are asserted rather than inferred from a label.
Because `Brain.compose` has no precision argument, the float32 worker rebuilds
its graph with `NeuralGraph(precision="float32")` before any learning or live use.
That setup is included in construction and memory costs on both revisions.
MPS requires Apple hardware and is an explicit unsupported-host skip here.

## Cost accounting

Each live sample is a synchronized complete batch decision, including sensory
construction, actual feedback, memory, declared teaching, free solve and action
readout. Its latency is **not divided by the number of rows**. Report throughput
in row decisions per second alongside whole-batch p50/p95/p99, maxima and counts
above a declared 50 ms diagnostic threshold; that threshold is not a real-time
guarantee. Cold imports/device initialization, construction, first query,
acquisition, held-out scoring and save/resume are separate timed stages. Parent
process wall time additionally includes imports, instrumentation and output I/O.

Work counters intercept solves without materializing device state. They include
completed settling calls, sweeps, row-sweeps and residual checks; exceptions have
their own attempted-call count. Partial sweeps inside an exception are unknown,
but the entire failed wall time remains charged. Free-answer diagnostics alone
would omit learning, eligibility, memory and other queries.

Host memory is the OS process-lifetime peak resident working set, including
imports and receipt collection. CUDA peaks are PyTorch allocated and reserved
bytes, including construction; they exclude driver/context and other processes.
Report these distinct scopes rather than calling allocator bytes total VRAM.
Transport cases reset allocator peaks before construction; host peaks remain
cumulative within that worker. No electrical-energy or hard real-time claim is
made. Profiler traces are diagnostic and are not used for throughput claims.

## Change review

- Minimalism: one CUDA block accumulation uses the existing transport and block
  order; no public API, learning rule, tolerance or dependency is added.
- User-friendliness: existing construction, query, teaching and checkpoint APIs
  retain their behavior, with explicit unsupported hardware and measured limits.
- Agent-friendliness: source/device/dtype, attempted work, refusal, source custody
  and continuation are inspectable and checked independently of timing claims.
