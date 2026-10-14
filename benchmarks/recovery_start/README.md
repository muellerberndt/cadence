# Where qualified nudged phases start settling

**Superseded.** `RecoveryStart` was removed after this measurement: its fitted
gains did not beat the unfitted copy, and no start helped on the composed
dynamics. The follow-up is [skipping the undamped attempt](../nudged_settle/README.md),
which keeps only the unfitted copy, as an agreement probe. This harness needs the
library source of commit `968e087`, which still has `RecoveryStart`. The receipt
and protocol are retained unchanged.

This paired CPU measurement asks one numerical question: if the qualified nudged
phases start from an estimate of their settled state instead of the free state,
do they qualify in fewer sweeps while reaching the same contrast? The library
option it tested was `Learner(recovery=RecoveryStart())` (commit `968e087`).
The equations, tolerance, qualification, damping schedule and contrast rule are
unchanged; only the starting state differs. It is not a learning-rule change and
makes no acquisition claim.

```sh
OMP_NUM_THREADS=1 python benchmarks/recovery_start/run.py \
  --out /tmp/recovery-start.json
```

`protocol.json` was written before the confirmation run and freezes the arms,
both nulls, the promotion criterion, seeds, cases and lesson counts. The school
rows are the [acquisition fixture](../acquisition/README.md)'s 24 Castlevania
sensory/action relations (650 inputs, 36 actions).

## Protocol

For each case and founder seed (1, 2, 3), a fresh `Brain.compose` brain is taught
16 qualified control lessons of four school rows. Then 12 measured lessons follow.
Each settles one qualified free phase and then both nudged phases once from each
arm's start on the same parameters, drive and labels. Only the default arm's
update is applied, so every arm sees the same acquired brain at every lesson. The
copy and fitted starts fit only from their own accepted phases. `sweeps` are the
learner's own qualified-phase counts (32-sweep residual chunks).
`sweeps_to_tolerance` re-solves the same start with a check after every sweep.
Wall time includes rendering the start. Trajectory cases also teach 24 identical
lessons to two fresh founders, with and without the fitted start, and compare the
learned parameters and free answers on all 24 rows.

| Arm | Start of both nudged phases |
| --- | --- |
| `default` | the free state (released behavior) |
| `copy` | `RecoveryStart(fit=False)`: the nudge copied down the seams once, every gain one |
| `fitted` | `RecoveryStart()`: decay 0.9, ridge 0.3, the only setting tried |
| `mirror` | positive phase from the free state; negative phase from the settled positive change reflected through the free state |

| Case | Modules | Motor lateral | Tolerance | Budget |
| --- | --- | ---: | ---: | ---: |
| `composed_default` | (32, 16) | -0.5 | 0.003 | 4096 |
| `lateral0` | (32, 16) | 0 | 1e-6 | 4096 |
| `deep_lateral0` | (32, 32, 32, 16) | 0 | 1e-6 | 4096 |

`composed_default` keeps the acquisition microscope's qualified dynamics
(`Brain.compose` at `dt=1`, rate 0.5, momentum 0.9) at its 0.003 tolerance; its
budget is raised from 1024 to 4096 so no measured lesson refuses.

## Results

Receipt: [`results/confirmation-2026-10-08.json`](results/confirmation-2026-10-08.json).
Windows 11, Python 3.12.7, NumPy float64, one thread, 36 measured lessons per
case, no refused lesson in any arm.

| Case | Arm | Nudged sweeps per lesson | Ratio to default | ms per lesson | Max contrast relative difference |
| --- | --- | ---: | ---: | ---: | ---: |
| `composed_default` | default | 931.9 | 1 | 26.8 | |
| | copy | 930.1 | 0.998 | 26.5 | 0 |
| | fitted | 931.0 | 0.999 | 28.5 | 0 |
| | mirror | 928.3 | 0.996 | 26.5 | 0 |
| `lateral0` | default | 140.4 | 1 | 4.06 | |
| | copy | 129.8 | 0.924 | 4.72 | 3e-4 |
| | fitted | 131.6 | 0.937 | 4.69 | 2e-4 |
| | mirror | 148.4 | 1.057 | 4.02 | 2e-4 |
| `deep_lateral0` | default | 614.2 | 1 | 21.1 | |
| | copy | 576.0 | 0.938 | 21.0 | 3e-4 |
| | fitted | 608.0 | 0.990 | 21.9 | **0.42** |
| | mirror | 637.3 | 1.038 | 22.9 | **0.099** |

The declared promotion criterion is **not met**; neither null is rejected on
`composed_default`, and the fitted start does not beat the unfitted copy in any
case.

- **The composed default forgets its start.** Every arm returned bit-identical
  qualified states (contrast difference exactly 0). These phases first run the
  undamped `dt=1` attempt into its numerical orbit (see
  `tests/test_qualified_learning.py`), and the halved attempts then continue from
  that orbit. The start therefore cannot change the cost; the integration
  schedule dominates it.
- **Without lateral inhibition the start saves 6 to 8 percent of sweeps**, and
  the unfitted copy saves at least as much as the fitted gains. The rendering
  costs more time than it saves at these sizes: every start arm is slower than
  the default in wall time on `lateral0`.
- **A different stationary point.** On `deep_lateral0`, founder 3, two of 12
  lessons settled the fitted start to a different qualified state (contrast
  relative difference 0.42 and 0.21, state difference 0.07). The mirror control
  did the same on two lessons (0.07 and 0.10). Both were qualified at 1e-6: the
  nudged equations have more than one stationary point there, and the default
  start selects one of them. Founder 3's output shell was poorly predicted
  (unexplained fraction 0.87 for gains, 0.85 for copy), so its start moved far.
- **Trajectories agree where the start is near.** Over 24 identical lessons, the
  final efficacies differ by at most 7e-7 relative (`lateral0`) and 7e-17
  (`composed_default`), with identical free answers on all 24 rows. Nudged sweeps
  fell 18, 13 and 3 percent on `lateral0` and under 1 percent on
  `composed_default`; wall time rose in all six.

Per-shell fidelity (fraction of the settled potential change left unexplained,
gains held before each lesson / every gain at one) shows the fitted gains help
only below the outputs, e.g. `deep_lateral0` shell 1: 0.13 / 0.27, 0.12 / 0.27,
0.10 / 0.48. At the output shell the gains are slightly worse than the copy in
every founder.

## Development runs

Before the protocol was written, an unretained prototype of the same start was
run on founder seed 0 with 6 to 12 lessons each. At the composed defaults the
public sweep counts were identical for every start, which motivated the orbit
explanation above. With lateral 0 and a one-sweep residual check, sweeps fell
from 81.8 to 74.6 (positive) and 87.0 to 81.1 (negative) for (32, 16), and from
120.3 to 111.4 (positive) with no negative-phase saving for (32, 32, 32, 16).
These were exploratory and are not evidence.

## Scope

This is three founders on one school, batch four, CPU float64. It measures sweep
counts and wall time of the nudged phases only; free phases, calibration and
memory are untouched. It does not test accelerator backends, larger graphs or
`ActorCritic` eligibility, which keeps its finite phases. Whether a seam start
helps graphs whose settle is a slow contraction rather than an orbit remains open.
