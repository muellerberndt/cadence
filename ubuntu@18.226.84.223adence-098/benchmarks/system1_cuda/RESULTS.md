# NVIDIA System 1 results — 2026-10-04

Direct CUDA block accumulation removes intermediate products and separate additions while preserving the existing equations, block order, tolerances and continuation. All 30 frozen campaign jobs passed; the independent verifier passed 686 checks. Four reversed-order confirmation jobs also passed. There were no nursery free-answer refusals, failed solve calls or finite-phase cap warnings.

A subsequent maintainer-style audit tightened the verifier and passed 1,041
checks on the same retained campaign. The original measurements and receipt
are preserved; see the audit below for the defects and their regression tests.

CPU with Numba remains faster than CUDA on this small continuing nursery. CUDA timing varies substantially on this Windows laptop: the results support reduced dispatch work and gains in these measured cases, not a general speedup, a real-time deadline or universal accelerator support.

## Sources and scope

- Baseline: `5cd3b3dbe3c30be8a39854a23a834950f705d366`.
- Candidate: `81a14cf1632218a07c0d3d16f70abdc27455c23a`; only `src/cadence/brain.py` differs among implementation sources.
- Hardware: NVIDIA RTX 4000 Ada Generation Laptop GPU, 12,282 MiB; Windows; driver 595.95; PyTorch 2.11.0+cu128 / CUDA 12.8; Python 3.13.2; NumPy 2.5.3; Numba 0.68.0; one host thread; TF32 disabled.
- CUDA float64 and float32 state execution were asserted on device 0. Parameters/optimizer retain the existing float64 contract. MPS was unavailable on this host.
- [Frozen protocol](protocol.json), [reproduction and accounting](README.md), and [compressed complete scalar record](results/rtx4000-ada-2026-10-04.json.gz). The record includes source/harness hashes, every raw timing, quality reports, profiles, confirmations and failed development checks. Binary state comparisons and raw trace hashes are retained; full local traces/checkpoints can be regenerated with the collector.
- This is the System 1 neural graph. Issue #64’s population-solver measurements are separate. Application access is now available, with supplementary compatibility checks below. The integrated Fable sequence benchmark still awaits the measured mechanism and dependency checks in [#121](https://github.com/muellerberndt/cadence/issues/121). No sequence capability or teacher-free speech-inference claim is made.

## Behavior before speed

Each of three founders acquired 100% held-out graph accuracy and retained 100% after the live sequence, on every backend/revision. These are memory-free graph controls. The same acquired brain then made 176 batches of 16 actions with live memory and actual feedback; its action accuracy below is lower. All baseline/candidate actions, update/memory counters and solve work were identical within each backend and seed. Backend-to-backend rounding can still change later actions.

| Live stage | CPU | CUDA float64 | CUDA float32 |
| --- | ---: | ---: | ---: |
| stable | 64.81% | 64.81% | 64.68% |
| disturbance | 64.97% | 64.97% | 64.19% |
| repair | 77.28% | 77.15% | 75.65% |
| continued | 92.02% | 91.93% | 92.42% |

Accuracy is pooled over three founders and is identical before/after the optimization. Disturbance reduces sensory amplitude; repair uses scheduled teacher labels. Continued behavior uses fresh observations without teacher labels, while actual-reward updates continue. This does not establish automatic failure-only repair or a general memory benefit. Saved continuation matched on every run.

## Complete costs and variability

Times below are seconds, before → after. Initialization includes imports/device startup and collector setup. Live totals cover all 176 complete batch decisions, including feedback, memory and repair. Process wall time additionally includes acquisition, scoring, checkpoint work, initialization and collection. Values are medians over three founders; raw per-run values remain in the record.

| Backend | Initialization | Construction | Live stages | Whole process | Live speedup range |
| --- | ---: | ---: | ---: | ---: | ---: |
| cpu | 0.863 → 0.718 | 0.010 → 0.011 | 1.012 → 0.975 | 3.530 → 3.410 | 0.99–1.34× |
| cuda64 | 4.022 → 3.218 | 0.088 → 0.098 | 16.018 → 11.947 | 32.965 → 24.359 | 1.16–3.20× |
| cuda32 | 3.187 → 3.201 | 0.103 → 0.100 | 44.374 → 13.982 | 62.090 → 27.246 | 1.06–3.17× |

CPU code is unchanged; its apparent timing changes illustrate run variability. Its first baseline process also populated a fresh Numba disk cache, so cold CPU differences do not establish an initialization or memory improvement. GPU results are from one laptop without isolated clocks/load; no statistical population claim is made.

The two large GPU timing differences were repeated with pair order reversed, preserving the same frozen protocol and all primary runs:

| Confirmation | Live seconds before → after | Live ratio | Whole-process seconds before → after |
| --- | ---: | ---: | ---: |
| cuda32, seed 1 | 21.819 → 15.459 | 1.41× | 38.968 → 37.115 |
| cuda64, seed 2 | 21.564 → 20.501 | 1.05× | 38.850 → 38.239 |

In particular, the largest original ratios did not repeat. No run was removed. All confirmation actions, work, source hashes and quality gates matched their original case.

## Decision latency and throughput

Pooled primary-run latency across the three founders, in milliseconds, for complete batches of 16. Latency is never divided by batch size; throughput counts individual row decisions. The 50 ms column is diagnostic misses before → after, not a deadline guarantee.

| Backend / stage | p50 before → after | p95 before → after | p99 before → after | Decisions/s before → after | >50 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| cpu / stable | 3.8 → 3.8 | 7.4 → 8.3 | 9.7 → 10.1 | 2412.4 → 3777.0 | 1 → 0 |
| cpu / disturbance | 3.7 → 3.8 | 6.5 → 6.8 | 7.8 → 8.1 | 4124.3 → 3905.7 | 0 → 0 |
| cpu / repair | 5.8 → 6.3 | 8.9 → 11.7 | 11.2 → 13.0 | 2709.2 → 2395.7 | 0 → 0 |
| cpu / continued | 4.2 → 4.0 | 8.6 → 8.3 | 12.8 → 12.5 | 3308.3 → 3381.5 | 0 → 0 |
| cuda64 / stable | 87.4 → 57.8 | 283.1 → 226.9 | 310.7 → 256.7 | 115.7 → 191.5 | 181 → 130 |
| cuda64 / disturbance | 87.6 → 55.6 | 277.8 → 97.1 | 288.5 → 119.9 | 115.2 → 267.3 | 44 → 32 |
| cuda64 / repair | 143.0 → 86.5 | 432.5 → 150.1 | 468.4 → 165.3 | 71.9 → 168.1 | 96 → 96 |
| cuda64 / continued | 88.9 → 58.0 | 286.9 → 105.4 | 297.6 → 119.0 | 115.6 → 259.5 | 184 → 133 |
| cuda32 / stable | 223.4 → 70.5 | 266.0 → 211.8 | 275.5 → 230.7 | 91.7 → 163.0 | 181 → 156 |
| cuda32 / disturbance | 227.1 → 64.6 | 273.9 → 98.7 | 283.9 → 110.9 | 86.3 → 233.2 | 45 → 40 |
| cuda32 / repair | 371.7 → 114.8 | 451.2 → 179.8 | 462.8 → 207.9 | 53.1 → 129.7 | 96 → 96 |
| cuda32 / continued | 225.3 → 65.3 | 268.9 → 126.0 | 280.7 → 147.5 | 89.6 → 224.7 | 187 → 157 |

## Peak memory

MiB. Host values are process-lifetime resident-set ranges across founders; CUDA values are allocator peaks (allocated / reserved), excluding driver/context and other processes. They include startup and saved-continuation work. The optimization did not lower the measured whole-life CUDA allocator peak.

| Backend | Host before | Host after | CUDA before | CUDA after |
| --- | ---: | ---: | ---: | ---: |
| cpu | 119.4–189.7 | 119.2–119.4 | — | — |
| cuda64 | 996.2–996.3 | 996.1–996.6 | 13.43 / 28.00 | 13.43 / 28.00 |
| cuda32 | 1063.2–1064.1 | 1063.5–1064.0 | 11.56 / 26.00 | 11.56 / 26.00 |

## Fixed-work and profiler controls

The control always performs 32 sweeps, from rest, with an additional 8-neuron observer; it is not an acquisition result. Table entries are median milliseconds before → after over 20 warmed samples. All raw tails, residuals and construction/memory costs are in the record.

| Modules / batch | CPU | CUDA float64 | CUDA float32 |
| --- | ---: | ---: | ---: |
| [64] / 1 | 0.30 → 0.16 | 68.06 → 56.26 | 77.04 → 65.81 |
| [64] / 16 | 1.60 → 1.38 | 80.71 → 54.42 | 73.23 → 60.97 |
| [64] / 64 | 6.22 → 6.65 | 67.30 → 51.37 | 85.40 → 58.55 |
| [256, 128] / 1 | 1.20 → 1.10 | 103.15 → 76.07 | 258.75 → 74.84 |
| [256, 128] / 16 | 11.63 → 11.89 | 81.51 → 58.38 | 224.83 → 70.80 |
| [256, 128] / 64 | 51.51 → 40.80 | 90.58 → 61.49 | 243.84 → 84.26 |

Maximum observed absolute potential deviation among all before/after and CPU-reference comparisons was 5.82e-06. Every comparison passed the predeclared absolute/relative tolerance (float64 1e-10; float32 2e-5). Those comparison tolerances do not change answer admission.

Profiling before the edit identified block transport as the main cost. The pinned float64 control performs the same 973 transport calls across its eight profiled decisions; one additional CUDA-profiled decision replaces 2,288 `mm` + `add_` + slice-writeback operations with 2,288 `addmm_` calls. Indexing (99 calls) and conversion (`_to_copy`, 74 calls) remain unchanged. This is an arithmetic/dispatch optimization, not a transfer-elimination claim. Larger batches amortize some launch cost; the declared layouts do not establish a universal GPU crossover.

The coupled profiler control uses finite teaching and records its capped-free-phase warning on both revisions. It is a bottleneck diagnostic, not evidence of qualified acquisition. Profiler wall times include instrumentation and are not the throughput table.

## Validation and remaining boundary

The actual-device tests independently check projected neuron equations, signed/zero nudges, masks, adaptation, sparse/blocked execution, observer feedback, overflow refusal, frozen parameters, positive-budget refusal, optimizer/witness custody, private imagination and checkpoint transfer. Allocation failure is deterministic fault injection after real transport, not physical VRAM exhaustion. Independent Newton finite differences cover quadratic and cross-entropy learning on CUDA float64/float32.

The compressed record preserves failed development fixtures and their eventual fixes. Some early collector failures did not record complete wall/VRAM costs; these unknowns remain explicit and are excluded from speed claims. The primary campaign cost 789.4 process-seconds; confirmation added 153.2 seconds. Neither campaign had a failed job.

Matched-workload MPS performance, integrated sequence timing, and the Fable audio/teacher-alignment/replay/inference-versus-update campaign are outstanding boundaries. The sequence item is blocked by the missing measured #121 integration and its explicit training hold, rather than repository access. This PR does not close that appended acceptance item or infer it from the nursery.

### Repository and installed-package checks

- The complete hardware-enabled run had 2,801 passes, 84 skips, one expected failure, and two pre-existing Windows provenance failures, reproduced on unchanged main with the baseline package imported explicitly. Both are fixed: POSIX receipt keys and LF checkout for a byte-hashed fixture. All 22 affected tests then passed; no source hash or qualification check was weakened.
- The final complete CPU reference run (CUDA explicitly masked with `CUDA_VISIBLE_DEVICES=-1`) reports 2871 cases, 0 failures, 0 errors, 180 skipped/expected-failure cases. This supplements the hardware run; its CUDA skips do not count as GPU qualification.
- Lint passes for source, tests, the collector and the two corrected examples. New Python files pass formatting. `mypy --platform linux` passes all 54 modules. Native Windows mypy has the same five `timing.py` platform-attribute errors as main; repository-wide format checking flags the same 69 existing files on both revisions. Those baseline findings remain explicit.
- The installed wheel passes the required 20 documentation pages and all three examples in a fresh NumPy-only environment. The independently installed source archive passes the quickstart and all three examples. Both pass `pip check`; optional backends are absent, and every installed implementation source matches the tested source after LF normalization. Artifact hashes and outcomes are in the compressed record.

### Checks after application access became available

The original compressed campaign record is unchanged. These later checks use
`cadence-transcribe` revision `e6bd446bf6812f1d486c007a398eabc24451f530` and the
same numerical sources as the campaign. Its complete existing suite passes all
40 tests on each of the pinned 0.71.1 release, baseline `5cd3b3d`, and PR
`3d2c81a`. These are compatibility results, not acquisition measurements.
Latest main `20ddde7` adds documentation only and has been integrated.

An additional fixed regression uses the application's eight WAV fixtures,
one untrained founder, modules `(16, 8)`, 1,960 sensory values, 36 labels,
explicit lateral weight -0.5, fixed zero/unit feature statistics, and one thread.
Each of CPU, CUDA float64 and CUDA float32 runs both revisions in fresh
processes, first querying all eight clips to warm up, then querying them again.
Preprocessing, solving, ranking, synchronization and refused work are timed.
Backend selection is explicit in the harness; application defaults are unchanged.

| Backend | Eight timed queries, before → after (s) | Whole process, before → after (s) |
| --- | --- | --- |
| CPU | 0.0417 → 0.0389 | 2.82 → 2.78 |
| CUDA float64 | 17.64 → 10.79 | 37.66 → 28.35 |
| CUDA float32 | 10.31 → 7.74 | 29.34 → 21.20 |

Every revision/backend has the same work counts and qualification outcomes:
six clips qualify and two refuse per pass, with zero of eight correct top-one
answers from these untrained brains. The maximum before/after activation
difference is 4.44e-16. These one-pair diagnostic timings include unsuccessful
work; they are not a speech speedup or audio-seconds/second qualification.
CPU is faster for this small batch-one control.

All six runs also preserve exact same-backend parameters and learning reports
after save/reload and one fixed synthetic lesson, then preserve the complete
brain checkpoint after a deliberately refused qualified lesson. Audio fixtures
are not used for teaching. The synthetic accepted lesson uses the application's
finite contract, not a qualified-learning certificate. The
[supplementary receipt](results/consumer-compatibility-2026-10-04.json) records
source identities, test counts, cold and warm timings, work, allocator peaks
and hashes of the local private-consumer records. Private application source,
audio and checkpoints are not copied into this public repository.

The initial macOS CI run also exposed two mistakes in the new tests: comparing
`mps` to the concrete device `mps:0`, and comparing stored float32 frozen
parameters to their pre-upload float64 values. Commit `a4d7ba7` checks the actual
device and exact stored parameters. Runtime equations and tolerances are
unchanged. Local CPU/CUDA revalidation passes 36 tests, with 12 unavailable-MPS
skips; the original failed [CI run](https://github.com/muellerberndt/cadence/actions/runs/37175424763)
remains available. The corrected checks also pass on the hosted macOS MPS
device in [Python 3.11](https://github.com/muellerberndt/cadence/actions/runs/37176167170/job/111359098830)
and [Python 3.13](https://github.com/muellerberndt/cadence/actions/runs/37176167170/job/111359098761).
This adds MPS correctness evidence, not a matched MPS performance measurement.

### Maintainer-style audit

Review followed the source-binding feedback on [PR #71](https://github.com/muellerberndt/cadence/pull/71#issuecomment-5924772369),
the effective-configuration and complete-work checks in [PR #130](https://github.com/muellerberndt/cadence/pull/130),
and the scoped numerical/behavioral claims of [PR #134](https://github.com/muellerberndt/cadence/pull/134)
and issues #98, #99 and #110. All 114 implementation, harness and protocol hashes
reproduce from their declared Git commits after the documented LF normalization.

The audit reproduced a verifier defect: duplicate jobs, incorrect device labels,
weakened reported tolerance, contradictory admission diagnostics and changed
actions could still receive `passed: true`. The underlying recorded campaign
was consistent; the verifier was incomplete. It now checks the ordered census,
actual execution metadata, effective learner/actor settings against both the
protocol and acquired checkpoint, per-row residuals and scores, complete latency
stages and deadline misses, and paired live actions and work. Twelve regression
tests use a pair from the published receipt and reject deliberate alterations.

Reverification of the retained 30-job campaign passes all 1,041 checks. The
[audit receipt](results/audit-2026-10-04.json.gz) retains the reproduced failures,
committed-byte checks, verifier identity and new verification outcomes. No
runtime, workload, tolerance, raw timing or original receipt was changed during
this audit. The supplementary private-consumer timings remain local diagnostics;
their private raw records and producer are not distributed by this PR.

This is bounded hardware evidence. It does not complete matched MPS performance,
physical allocation-failure/resource-envelope qualification in #99, general
acquisition/retention in #110, or the pending #121/Fable sequence workload.
