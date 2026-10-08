# Skipping the undamped attempt of qualified nudged phases

This paired CPU measurement follows the [recovery-start measurement](../recovery_start/README.md).
That measurement found that on the composed brain's dynamics every nudged phase
forgets where it started: the undamped `dt=1` attempt of the `damping` schedule
spends its whole budget share in an orbit, and only the halved attempts settle.
Here the question is whether starting the nudged phases further down the same
schedule settles them in fewer sweeps while reaching the same qualified result.
The library option it tests is `Learner(nudged_settle=NudgedSettle())`
([learning guide](../../docs/learning.md#skipping-the-undamped-attempt-of-nudged-phases)).

```sh
OMP_NUM_THREADS=1 python benchmarks/nudged_settle/run.py --out /tmp/nudged-settle.json
```

`protocol.json` was written before the confirmation run. It freezes the arms, both
nulls, the promotion criterion and the reference solve. The founders, school rows,
cases, acquisition and lesson sequence are those of the recovery-start protocol:
three founders (seeds 1 to 3), 16 acquisition lessons, then 12 paired lessons of
four school rows. Each paired lesson settles one qualified free phase, then both
nudged phases once per arm on the same parameters, drive and labels, plus a tight
reference solve (tolerance 1e-10, `dt / 16` to `dt / 256`). Only the default arm's
update is applied. The trajectory cases teach 24 identical lessons to two fresh
founders, one with the default learner and one with `NudgedSettle()`.

| Arm | Nudged phases |
| --- | --- |
| `default` | released schedule (`damping=3`: `dt`, `dt/2`, `dt/4`, `dt/8`) |
| `halve1` | `first_halving=1`: `dt/2`, `dt/4`, `dt/8`; no probe |
| `halve2` | `first_halving=2`: `dt/4`, `dt/8`; no probe |
| `candidate` | `NudgedSettle()`: `halve2` plus the agreement probe and default fallback, all work charged |
| `local_step` | a per-neuron step `1 / (1 + g_i)`, `g_i` the presynaptic-slope-weighted sum of a neuron's incoming `|W|` at the free state; harness-only NumPy, residual checked every 32 sweeps |

`local_step` is the per-patch step size used by the external work that motivated
this measurement. Its fixed point is the model's own, but it is not part of the library.

## Results

Receipt: [`results/confirmation-2026-10-08.json`](results/confirmation-2026-10-08.json).
Windows 11, Python 3.12.7, NumPy float64, one thread, 36 measured lessons per case,
no refused lesson. Sweeps include probe and discarded work. Contrast error is the
relative distance of the lesson's synaptic contrast from the tight reference's.

| Case | Arm | Nudged sweeps per lesson | Ratio to default | ms per lesson | All qualified | Contrast error max / mean | Fallback phases |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| `composed_default` | default | 931.9 | 1 | 51.0 | yes | 0.260 / 0.074 | |
| | halve1 | 678.7 | 0.728 | 36.0 | yes | 0.055 / 0.017 | |
| | halve2 | 64.9 | 0.070 | 5.8 | yes | 0.191 / 0.116 | |
| | **candidate** | **128.9** | **0.138** | **14.0** | yes | **0.191 / 0.116** | 0 |
| | local_step | 105.8 | 0.114 | 95.0 | yes | 0.272 / 0.188 | |
| `lateral0` | default | 140.4 | 1 | 8.5 | yes | 2e-4 / 1e-4 | |
| | halve1 | 250.7 | 1.785 | 15.2 | yes | 5e-4 / 1e-4 | |
| | halve2 | 477.3 | 3.399 | 26.1 | yes | 5e-4 / 1e-4 | |
| | candidate | 931.6 | 6.633 | 53.8 | yes | 5e-4 / 1e-4 | 0 |
| | local_step | 1239.1 | 8.823 | 988.9 | yes | 3e-4 / 1e-4 | |
| `deep_lateral0` | default | 614.2 | 1 | 43.3 | yes | 0.0016 / 2e-4 | |
| | halve1 | 1158.3 | 1.886 | 80.3 | yes | 0.0016 / 3e-4 | |
| | halve2 | 2340.4 | 3.810 | 156.6 | **no** | 0.0067 / 5e-4 | |
| | candidate | 4392.9 | 7.152 | 304.9 | yes | 0.0018 / 3e-4 | 3 |
| | local_step | 4232.9 | 6.891 | 3538.7 | **no** | 0.059 / 0.0025 | |

Trajectories, 24 lessons per founder (nudged sweeps; complete-lesson wall time,
including free phases):

| Case | Founder | Default | Candidate | Efficacy rel. difference | Free answers differing (of 24) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `composed_default` | 1 | 25,260; 2.06 s | 3,072; 1.04 s | 6.5e-4 | 5 |
| | 2 | 22,252; 1.82 s | 3,072; 0.95 s | 1.1e-3 | 4 |
| | 3 | 20,928; 2.09 s | 3,072; 1.19 s | 1.2e-3 | 4 |
| `lateral0` | 1 | 2,688; 0.29 s | 16,320; 1.14 s | 4.3e-7 | 0 |
| | 2 | 2,688; 0.27 s | 16,000; 1.14 s | 8.1e-7 | 0 |
| | 3 | 7,168; 0.62 s | 51,232; 2.97 s | 1.1e-6 | 0 |

### Against the declared criterion

- **N1 rejected:** on the composed dynamics the candidate charged 13.8 percent of
  the default's nudged sweeps (7.2 times fewer), and nudged-phase time fell from
  51 to 14 ms per lesson. All phases qualified, no probe disagreed, and nothing
  fell back. Whole lessons, including the unchanged free phase, took half the time.
- **N2 rejected on the declared statistic:** the largest per-lesson contrast error
  to the tight reference was 0.191 for the candidate against 0.260 for the
  default. The mean was worse, 0.116 against 0.074. At tolerance 0.003 both paths
  are within the qualification error of the same equilibrium; neither tracks the
  tight solve closely.
- **Promotion criterion not met:** the trajectories did not keep identical free
  answers. 4 or 5 of 24 differed per founder. A post-hoc check
  ([`posthoc_tight.py`](posthoc_tight.py), declared after this run) repeated the
  three trajectories at tolerance 1e-8. Default and candidate each matched the
  tight trajectory on 65 of 72 answers (21/24/20 and 21/20/24). The disagreement is
  the tolerance's own drift, which both paths share, rather than the candidate
  leaving the default's equilibrium. The frozen criterion still fails.
- **The step change is not general.** Where the full step already settles
  (`lateral0`, `deep_lateral0`), every smaller first step costs 1.8 to 7 times
  more sweeps, and `halve2` exhausted the budget on some deep lessons, which the
  candidate's fallback caught (3 phases). The option is for graphs whose undamped
  attempt orbits, not a default.
- **`halve1`** is the closest arm to the tight reference on the composed dynamics
  (contrast error 0.055 / 0.017) and still saves 27 percent.
- **The per-neuron step did not beat the uniform one.** `local_step` settled the
  composed phases in 106 sweeps against `halve2`'s 65 and was slowest in time (dense
  NumPy). It was the slowest arm, or failed to qualify, without lateral
  inhibition. It is not proposed for the library.

## Scope

This covers three founders on one school, batch four and CPU float64 only, and
only the nudged phases. The free phase pays the same orbit: a development check on
founder 0 needed 739 free sweeps under the default schedule and 53 starting at
`dt / 4`. That is not part of this option. Neither are `act`, `predict` and the
reward eligibility path. Whether an adaptive schedule could abandon an orbiting
attempt early while keeping the full step where it settles remains open.
