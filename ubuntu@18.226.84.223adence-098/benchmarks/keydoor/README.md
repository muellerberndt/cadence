# Key-door nursery

This bounded instrument asks whether one continuing `Brain.compose` life can credit an
action whose worth arrives several irrelevant choices later, keep exploring at a cost
when its food stops, and move its search when the source of the food moves. It is the
delayed key-door reward nursery of
[issue 111](https://github.com/muellerberndt/cadence/issues/111), roadmap row 07 of
[issue 109](https://github.com/muellerberndt/cadence/issues/109), built on the routine
and repair loop of [issue 122](https://github.com/muellerberndt/cadence/issues/122) and
modelled on the [odour nursery](../reversal/README.md) of row 05. It measures one declared
System 1 operating point with arousal against the same brain without it, the brain without
eligibility, with yoked rewards, frozen, without its pouch sense, a tabular learner with the
same information and uniform-random actions. It establishes no default. Read
[routine and repair](../../docs/start.md#routine-and-repair-live), the
[world-model guide](../../docs/start.md#one-continuing-equilibrium-brain) and
[numerical contracts](../../docs/contracts.md) before interpreting results.

## Evidence status after review

All historical protocols and compressed receipts remain byte-for-byte unchanged. The two
`key-door/1` freezes and the `key-door/3` freeze failed their declared gates. Their tables
below describe their pinned sources. The `/1` world coupled action-dependent food draws,
cuts and yoked placement; `/2` separated those streams and repaired the endpoint probes,
cut range, door-opening window and work accounting.

The current `key-door/3` instrument has `instrument_revision: 2`. Maintainer review found
that the recurrent control combined activations from before a feedback update with weights
from after it, omitted the actor readout from its gradient norm, and did not clip critic
gradients. The corrected control chooses and differentiates one policy after feedback and
bounds each complete actor and critic gradient. This changes the recurrent control; the
historical recurrent results are not results of the correction. Cadence's runtime,
operating point and defaults are unchanged.

Revision 2 records completed-trip outcomes and moment sums so verification can recompute
feeding, cost, return lag, arousal and policy summaries. It counts every arm's action and
probe calls, conventional updates and parameters, gradient clips and failed-call time.
Its separate joint audit requires the same lives to meet all criteria and the complete
unchanged confirmation census; development, selected arms, selected delays and overrides
cannot pass. Historical marginal gate labels remain historical. An audit of already spent
seeds is not a fresh confirmation, and issue 111 remains open.

## Corrected recurrent-control audit, 2026-10-08

The [corrected receipt](results/audit-recurrent-maintainer-2026-10-08.json.gz) and
[verification counts](results/verification-recurrent-maintainer-2026-10-08.json) use
source `abbe9adb`, the unchanged declared recurrent settings, and all 30 recurrent lives
on already spent seeds 900–909. No Cadence runtime or default changed. This control-only
audit is ineligible for the full confirmation gate and does not replace the historical freeze.

| Delay | A feeding, historical → corrected | B feeding, historical → corrected | Return feeding, historical → corrected | Corrected first-window counts A/B/return |
| --- | --- | --- | --- | --- |
| 2 | 0.960 → 0.956 | 0.886 → 0.884 | 0.994 → 0.996 | 10/10, 9/10, 10/10 |
| 5 | 0.960 → 0.956 | 0.684 → 0.684 | 0.986 → 0.988 | 10/10, 7/10, 10/10 |
| 10 | 0.762 → 0.956 | 0.380 → 0.484 | 0.890 → 0.982 | 10/10, 5/10, 10/10 |

The corrected control acquires at every delay; re-adaptation remains incomplete. In
particular, the old delay-10 acquisition failures were not reliable evidence about a
correctly differentiated control. The run completed without errors: 657,333 decisions,
657,303 feedback updates, 1,890 probe calls, 9,018 actor-gradient clips and no critic-gradient
clips, with 419 learned scalars per founder. Summed founder elapsed time was 162.90 seconds;
this is not a matched compute or energy comparison. The original source-bound verifier
accepted the complete receipt and recomputed its trip, moment and gate arithmetic.

## Maintainer development: stable skill and joint census, 2026-10-08

The original 20 gated founders meet all five predicates jointly in **11/20 live** and
**16/20 blind** lives, below the unchanged 90% bound. Live marginal counts are 16 acquired,
18 adapted, 16 frugal, 13 calm and 18 returned; blind counts are 20, 19, 20, 17 and 19.
The blind misses are all at delay 5: seed 903 returns at lag 68; seed 904 ends B fed on
0.66 of trips and late arousal 0.405; seeds 906 and 908 acquire late (lags 425 and 400),
with late arousal 0.653 and 0.527. These are audits of spent seeds, not new confirmation.

The separately declared [`protocol-retention-development.json`](protocol-retention-development.json)
and [`retention_audit.py`](retention_audit.py), run from source `abbe9adb`, measure a stable
problem shared by both rules: a door faced with a key. At birth, after A and after B, each
saved copy starts a fresh stream, retains learned weights and records, and makes one greedy
action without receiving feedback. The original life and its pending outcome remain intact.
This asks whether the usable door response survives competing experience; it does not ask a
frozen policy to infer the unannounced return to A.

Four development lives (live/blind, seeds 0 and 1, delay 5, 500 A trips then 500 B trips)
completed without refusals. All four fed on the private door probe after A and after B.
Three already fed on that probe at birth. Blind seed 1 changed from pass to interact after A
and preserved it after B; its interaction probabilities were 0.496, 0.941 and 0.959.
The continuing lives ended A/B fed at 1.00/1.00 and 0.96/0.86 for live, and 1.00/1.00
for both blind lives. This is bounded development evidence for one stable response;
it does not close the failed joint gate or establish preservation of an unobservable rule.

The [compressed receipt](results/development-retention-maintainer-2026-10-08.json.gz) and
[verification receipt](results/verification-retention-maintainer-2026-10-08.json) bind the
readings to `abbe9adb`; its original verification checked all 36 local checkpoint artifacts.
Later verifier-only edits change the current source manifest: `--current` for this historical
run requires its pinned source. The NPZ files remain in local
`runs/cadence-pr156-maintainer-20261008/retention/`. All original checkpoint state
and all private-copy learned state were preserved. Founder elapsed time totalled 251.51
seconds; there were 58,632 actual decisions. The assay charges its four Brain save/load
operations and greedy solve per boundary; state-file verification reads are included in
elapsed time. No energy or efficiency comparison follows.

```sh
python benchmarks/keydoor/retention_audit.py --out /tmp/keydoor-retention/receipt.json
python benchmarks/keydoor/retention_audit.py --verify /tmp/keydoor-retention/receipt.json --current --artifacts
```

The bounded follow-up in
[`protocol-eligibility-development.json`](protocol-eligibility-development.json) compared
blind-body founders 2 and 3 at delay 5 with the existing eligibility gene `lam=0.95` and
`0.98`. It ran from source `7487a031`, with all three 500-trip phases and every original
gate unchanged.
The [`eligibility_development.py`](eligibility_development.py) helper stopped after those
four lives. Both [control](results/development-eligibility-control-maintainer-2026-10-08.json.gz)
and [candidate](results/development-eligibility-candidate-maintainer-2026-10-08.json.gz)
receipts passed source and arithmetic verification; the
[verification record](results/verification-eligibility-maintainer-2026-10-08.json) preserves
the declaration, paired readings and work counts.

| Founder | A lag, 0.95 → 0.98 | B lag, 0.95 → 0.98 | Return lag, 0.95 → 0.98 | A late arousal, 0.95 → 0.98 |
| --- | --- | --- | --- | --- |
| 2 | 254 → 346 | 23 → 55 | 9 → 2 | 0.103 → 0.392 |
| 3 | 68 → 76 | 54 → 117 | 3 → 17 | 0.002 → 0.060 |

**The candidate is rejected.** Both founders acquired and adapted later; founder 2 missed
the unchanged calm bound of 0.35, and founder 3 returned later. Feeding stayed at 1.00 and
wrong interactions at 0.00 in every phase. All-five-predicate development counts fell from
2/2 to 1/2. Each setting used 43,750 decisions and 126 probes, without failures. Learning
sweeps rose from 69,774 to 107,832, while total charged action, learning and probe sweeps
were nearly equal (1,432,910 and 1,432,888). Founder elapsed time totalled 57.50 seconds
across both settings. This development changes no runtime or default, admits no confirmation
claim, and does not close issue 111.

The early `development-3-copy-amp3` receipt used another instrument: its arms were
`live` (with the copy), `nocopy` and `tabular`, with chamber hash `d1fd408a4600…`.
Its original chamber source has not been recovered. The current verifier admits only that
exact archived source manifest as **custody only** (canonical bytes, digest, protocol and
census), and does not reinterpret its arithmetic or gates. Every historical byte is retained.

## Historical key-door/3 freeze, 2026-10-08

The third freeze, [`protocol-3.json`](protocol-3.json), SHA-256
`f8bf4d7698c72905fca4e1384a4c07389ba87360b899177e19cc765208bf56c9`, committed before its
confirmation seeds 900 to 909 were run. It keeps key-door/2's corrected instrument and
operating point and adds, as roadmap row 07 asked: a third rule, the key back in the chest
after the lamp, to measure return/reacquisition of the first contingency (the historical “retained” gate:
at least 90% of the gated lives end that rule fed and find their first 20-trip window at
90% fed within 50 completed trips); lever counts varying by two from trip to trip in a
15-cell corridor, the irregular event time of the acceptance (physical time stays
unmodelled, as the rhythm chamber established per-event determinism for this library);
a `recurrent` arm, an online Elman actor-critic with eligibility traces and a gradient
guard, a recurrent online learner with the same information, its rate 0.02
and decay 0.8 selected on the development seeds; and a `copy` arm, the live brain carrying
the efference copy of its own last command (0.76.0) at a dose of 0.3 selected on the
development seeds, reported and not gated, because every dose harmed the creature there
(amplitude 3.0, the reward-rhythm chamber's, left every development life unfed and
restless; 1.0 fed 0.51 and 0.14 under rule A at delays 2 and 5; 0.3 fed 0.82 and 0.65
against the live brain's 0.90 and 0.99). The live arm therefore stays the simplest existing
System 1 at key-door/2's point, and the copy's founder value, zero, is that arm. The
development receipts are in `results/development-3-*`; the live arm on development seeds
0 to 7 acquired on 8 of 8 at both gated delays, re-adapted on 15 of 16 and took the
returned key back within a median of 6 and 13 completed trips against 23 and 125 at first
acquisition.

Receipt `results/confirmation-3-2026-10-08.json.gz`, bound to the source at
commit `6b6dc0e` (before maintainer corrections): 300 lives, none crashed, no refused answer. **The declared gates failed**, pooled
over delays 2 and 5: acquired 0.80, adapted 0.90, frugal 0.80, calm 0.65 and retained 0.90
against 0.90 required. Two lives per gated delay end rule A below 90% fed (minima 0.58
and 0.46), three lives at delay 2 waste more than 1.5 interactions per trip, and seven of
twenty spend more than 35% of a rule's second half aroused. The marginal re-adaptation and return
readings meet their historical bounds: the live brain ends the third rule fed in 0.99 of its last 50 trips
at both delays, close to the frozen rule-A policy's 1.00 and 0.98, and finds its window at a median of 17 and 12 completed trips. The third rule
allows another 500 learning trips; this is return/reacquisition evidence, not a test of
preserved skill without relearning. Delay 10
stays unsolved by the brain (0.49 fed) and solved by the tabular learner (0.83 to 0.90).

The controls say where the creature's credit comes from. The yoked arm, its own door
outcome paid at a random cell of the next trip, never acquires (0.15 to 0.22); the
eligibility-zero arm acquires at delay 2 (0.94) and fails from delay 5 (0.53); the
always-learning `step` loop acquires (0.93, 0.88) and re-adapts poorly (0.69, 0.51) at
every moment aroused. The copy arm is worse than the live arm at every delay except rule B
at delay 5 (0.93 against 0.92), and its calm is 0.47 and 0.33. The recurrent learner
acquires on every life at delays 2 and 5 (0.96, lags 171 and 187 against the brain's 21
and 15), re-adapts in 9 and 7 of 10 (0.89, 0.68) and returns on every life (0.99). The
tabular learner stays at 0.80 to 0.92 with its epsilon. The uniform-random policy feeds
0.23 to 0.30.

**The blind body has the strongest historical Cadence results at the gated delays.**
Without the pouch sense, the creature ends every rule
fed at 1.00, 1.00 and 1.00 at delay 2 and 1.00, 0.97 and 0.99 at delay 5, with 0.00 to
0.23 wrong interactions per trip and 3% to 8% of its late moments aroused; it even feeds
0.82 at delay 10 under rule A. The pouch bit tells the creature whether it holds the key;
a reactive policy over cell kinds alone can also succeed here. No working-trace ablation
is supplied, so the blind result does not establish memory of key possession. Its 16/20
joint result still misses the unchanged bound. The pouch-less body remains a candidate
for bounded development with the pouch body as its control; a new confirmation would
require a separately declared candidate that meets the original criteria in development.
This would change the world's body, an adapter choice, without changing the brain.

#### Rule A, chest holds the key: episodes fed in the last 50, mean (minimum)

| Arm | D=2 | D=5 | D=10 |
| --- | --- | --- | --- |
| `live` | 0.92 (0.58) | 0.92 (0.46) | 0.49 (0.00) |
| `copy` | 0.70 (0.50) | 0.66 (0.18) | 0.15 (0.02) |
| `step` | 0.93 (0.84) | 0.88 (0.62) | 0.58 (0.00) |
| `lambda-zero` | 0.94 (0.62) | 0.53 (0.00) | 0.03 (0.00) |
| `yoked` | 0.22 (0.14) | 0.15 (0.02) | 0.18 (0.04) |
| `frozen` | 0.92 (0.58) | 0.92 (0.46) | 0.49 (0.00) |
| `blind` | 1.00 (1.00) | 1.00 (0.96) | 0.82 (0.20) |
| `recurrent` | 0.96 (0.90) | 0.96 (0.90) | 0.76 (0.00) |
| `tabular` | 0.90 (0.86) | 0.80 (0.00) | 0.83 (0.00) |
| `random` | 0.26 (0.16) | 0.23 (0.10) | 0.23 (0.16) |

#### Rule B, lamp holds the key: episodes fed in the last 50, mean (minimum)

| Arm | D=2 | D=5 | D=10 |
| --- | --- | --- | --- |
| `live` | 0.97 (0.68) | 0.92 (0.24) | 0.41 (0.02) |
| `copy` | 0.73 (0.46) | 0.93 (0.58) | 0.17 (0.00) |
| `step` | 0.69 (0.00) | 0.51 (0.00) | 0.14 (0.00) |
| `lambda-zero` | 0.65 (0.02) | 0.13 (0.00) | 0.02 (0.00) |
| `yoked` | 0.12 (0.04) | 0.04 (0.00) | 0.05 (0.00) |
| `frozen` | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) |
| `blind` | 1.00 (1.00) | 0.97 (0.66) | 0.55 (0.02) |
| `recurrent` | 0.89 (0.00) | 0.68 (0.00) | 0.38 (0.00) |
| `tabular` | 0.92 (0.82) | 0.90 (0.84) | 0.90 (0.82) |
| `random` | 0.23 (0.10) | 0.25 (0.20) | 0.23 (0.14) |

#### Rule A: key taken in the last 50 episodes, mean (minimum)

| Arm | D=2 | D=5 | D=10 |
| --- | --- | --- | --- |
| `live` | 0.95 (0.72) | 0.94 (0.64) | 0.58 (0.14) |
| `copy` | 0.82 (0.66) | 0.77 (0.34) | 0.35 (0.12) |
| `step` | 0.95 (0.86) | 0.90 (0.68) | 0.59 (0.00) |
| `lambda-zero` | 0.97 (0.78) | 0.59 (0.04) | 0.14 (0.10) |
| `yoked` | 0.43 (0.32) | 0.39 (0.08) | 0.37 (0.14) |
| `frozen` | 0.95 (0.72) | 0.94 (0.64) | 0.58 (0.14) |
| `blind` | 1.00 (1.00) | 1.00 (0.96) | 0.85 (0.30) |
| `recurrent` | 0.98 (0.92) | 0.98 (0.92) | 0.80 (0.10) |
| `tabular` | 0.95 (0.92) | 0.85 (0.04) | 0.87 (0.02) |
| `random` | 0.51 (0.40) | 0.50 (0.34) | 0.48 (0.44) |

#### Rule B: key taken in the last 50 episodes, mean (minimum)

| Arm | D=2 | D=5 | D=10 |
| --- | --- | --- | --- |
| `live` | 0.98 (0.84) | 0.93 (0.36) | 0.47 (0.10) |
| `copy` | 0.80 (0.56) | 0.95 (0.64) | 0.30 (0.10) |
| `step` | 0.70 (0.00) | 0.51 (0.00) | 0.15 (0.00) |
| `lambda-zero` | 0.68 (0.08) | 0.28 (0.16) | 0.15 (0.10) |
| `yoked` | 0.35 (0.20) | 0.21 (0.12) | 0.20 (0.06) |
| `frozen` | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) |
| `blind` | 1.00 (1.00) | 0.97 (0.74) | 0.61 (0.16) |
| `recurrent` | 0.90 (0.00) | 0.69 (0.00) | 0.39 (0.00) |
| `tabular` | 0.97 (0.94) | 0.95 (0.90) | 0.95 (0.92) |
| `random` | 0.52 (0.42) | 0.52 (0.44) | 0.48 (0.36) |

#### Rule A: wrong interactions per episode in the last 50, mean (minimum)

| Arm | D=2 | D=5 | D=10 |
| --- | --- | --- | --- |
| `live` | 0.37 (0.00) | 0.40 (0.00) | 1.73 (0.00) |
| `copy` | 1.30 (0.00) | 1.64 (0.00) | 3.80 (1.36) |
| `step` | 0.94 (0.14) | 1.00 (0.10) | 0.79 (0.12) |
| `lambda-zero` | 0.54 (0.00) | 0.56 (0.00) | 1.66 (1.24) |
| `yoked` | 1.17 (0.72) | 1.84 (1.16) | 2.48 (1.02) |
| `frozen` | 0.37 (0.00) | 0.40 (0.00) | 1.73 (0.00) |
| `blind` | 0.19 (0.00) | 0.13 (0.00) | 0.59 (0.00) |
| `recurrent` | 0.09 (0.04) | 0.08 (0.02) | 0.14 (0.02) |
| `tabular` | 0.15 (0.04) | 0.70 (0.34) | 1.54 (0.46) |
| `random` | 1.52 (1.28) | 3.00 (2.68) | 5.57 (5.10) |

#### Rule B: wrong interactions per episode in the last 50, mean (minimum)

| Arm | D=2 | D=5 | D=10 |
| --- | --- | --- | --- |
| `live` | 0.43 (0.00) | 0.19 (0.00) | 1.27 (0.00) |
| `copy` | 1.28 (0.00) | 0.73 (0.00) | 3.19 (1.44) |
| `step` | 1.55 (0.12) | 0.77 (0.02) | 0.28 (0.06) |
| `lambda-zero` | 0.29 (0.00) | 0.81 (0.00) | 1.60 (1.32) |
| `yoked` | 0.94 (0.64) | 1.10 (0.76) | 2.14 (1.36) |
| `frozen` | 1.00 (1.00) | 1.00 (1.00) | 0.36 (0.00) |
| `blind` | 0.00 (0.00) | 0.06 (0.00) | 1.03 (0.00) |
| `recurrent` | 0.93 (0.04) | 0.83 (0.04) | 0.24 (0.00) |
| `tabular` | 0.16 (0.08) | 0.69 (0.46) | 1.74 (1.24) |
| `random` | 1.43 (1.26) | 2.94 (2.70) | 5.41 (5.14) |

#### Rule A: first 20-episode window 90% fed, median episode (lives / lives)

| Arm | D=2 | D=5 | D=10 |
| --- | --- | --- | --- |
| `live` | 21 (10/10) | 15 (9/10) | 298 (4/10) |
| `copy` | 126 (9/10) | 152 (5/10) | 305 (1/10) |
| `step` | 162 (10/10) | 170 (9/10) | 288 (8/10) |
| `lambda-zero` | 43 (10/10) | 32 (6/10) | 10 (2/10) |
| `yoked` | 16 (2/10) | 39 (2/10) | 195 (1/10) |
| `frozen` | 21 (10/10) | 15 (9/10) | 298 (4/10) |
| `blind` | 6 (10/10) | 9 (10/10) | 126 (8/10) |
| `recurrent` | 171 (10/10) | 187 (10/10) | 241 (8/10) |
| `tabular` | 16 (10/10) | 25 (9/10) | 28 (9/10) |
| `random` | none (0/10) | none (0/10) | none (0/10) |

#### Rule B: first 20-episode window 90% fed, median episode (lives / lives)

| Arm | D=2 | D=5 | D=10 |
| --- | --- | --- | --- |
| `live` | 42 (10/10) | 64 (9/10) | 147 (3/10) |
| `copy` | 125 (9/10) | 93 (9/10) | none (0/10) |
| `step` | 63 (7/10) | 80 (5/10) | 187 (2/10) |
| `lambda-zero` | 42 (6/10) | 44 (1/10) | none (0/10) |
| `yoked` | none (0/10) | none (0/10) | none (0/10) |
| `frozen` | none (0/10) | none (0/10) | none (0/10) |
| `blind` | 9 (10/10) | 30 (10/10) | 15 (6/10) |
| `recurrent` | 82 (9/10) | 125 (7/10) | 229 (4/10) |
| `tabular` | 16 (10/10) | 17 (10/10) | 29 (10/10) |
| `random` | none (0/10) | none (0/10) | none (0/10) |

#### Rule A behaviour: probability of taking at the chest without the key, median (minimum)

| Arm | D=2 | D=5 | D=10 |
| --- | --- | --- | --- |
| `live` | 0.961 (0.565) | 0.908 (0.447) | 0.403 (0.311) |
| `copy` | 0.728 (0.619) | 0.613 (0.414) | 0.420 (0.258) |
| `step` | 0.824 (0.587) | 0.792 (0.559) | 0.516 (0.219) |
| `lambda-zero` | 0.959 (0.533) | 0.643 (0.242) | 0.206 (0.186) |
| `yoked` | 0.486 (0.434) | 0.445 (0.400) | 0.408 (0.302) |
| `frozen` | 0.961 (0.565) | 0.908 (0.447) | 0.403 (0.311) |
| `blind` | 0.991 (0.813) | 0.986 (0.432) | 0.695 (0.353) |
| `recurrent` | 0.839 (0.815) | 0.796 (0.718) | 0.562 (0.149) |
| `tabular` | 0.932 (0.283) | 0.929 (0.087) | 0.892 (0.059) |
| `random` | 0.500 (0.500) | 0.500 (0.500) | 0.500 (0.500) |

#### Rule B behaviour: probability of taking at the lamp without the key, median (minimum)

| Arm | D=2 | D=5 | D=10 |
| --- | --- | --- | --- |
| `live` | 0.937 (0.764) | 0.868 (0.356) | 0.334 (0.153) |
| `copy` | 0.771 (0.477) | 0.852 (0.578) | 0.385 (0.148) |
| `step` | 0.743 (0.029) | 0.177 (0.013) | 0.017 (0.011) |
| `lambda-zero` | 0.772 (0.146) | 0.159 (0.148) | 0.144 (0.142) |
| `yoked` | 0.391 (0.244) | 0.273 (0.160) | 0.189 (0.149) |
| `frozen` | 0.000 (0.000) | 0.000 (0.000) | 0.000 (0.000) |
| `blind` | 0.976 (0.892) | 0.948 (0.518) | 0.570 (0.184) |
| `recurrent` | 0.789 (0.009) | 0.711 (0.005) | 0.014 (0.003) |
| `tabular` | 0.925 (0.873) | 0.920 (0.778) | 0.901 (0.170) |
| `random` | 0.500 (0.500) | 0.500 (0.500) | 0.500 (0.500) |

#### Rule B behaviour: probability of interacting at a lever with the key, median (minimum)

| Arm | D=2 | D=5 | D=10 |
| --- | --- | --- | --- |
| `live` | 0.084 (0.008) | 0.048 (0.009) | 0.176 (0.041) |
| `copy` | 0.554 (0.130) | 0.150 (0.014) | 0.378 (0.147) |
| `step` | 0.292 (0.112) | 0.035 (0.015) | 0.016 (0.011) |
| `lambda-zero` | 0.235 (0.006) | 0.157 (0.145) | 0.142 (0.141) |
| `yoked` | 0.333 (0.250) | 0.219 (0.157) | 0.182 (0.147) |
| `frozen` | none | none | none |
| `blind` | 0.009 (0.002) | 0.011 (0.002) | 0.088 (0.002) |
| `recurrent` | 0.020 (0.011) | 0.005 (0.004) | 0.004 (0.002) |
| `tabular` | 0.059 (0.056) | 0.127 (0.108) | 0.155 (0.115) |
| `random` | 0.500 (0.500) | 0.500 (0.500) | 0.500 (0.500) |

#### Rule A again, the key back in the chest: episodes fed in the last 50, mean (minimum)

| Arm | D=2 | D=5 | D=10 |
| --- | --- | --- | --- |
| `live` | 0.99 (0.90) | 0.99 (0.94) | 0.36 (0.00) |
| `copy` | 0.91 (0.60) | 0.84 (0.04) | 0.22 (0.00) |
| `step` | 0.98 (0.94) | 0.49 (0.00) | 0.29 (0.00) |
| `lambda-zero` | 0.71 (0.00) | 0.02 (0.00) | 0.02 (0.00) |
| `yoked` | 0.09 (0.02) | 0.03 (0.00) | 0.03 (0.00) |
| `frozen` | 1.00 (1.00) | 0.98 (0.82) | 0.38 (0.00) |
| `blind` | 1.00 (0.98) | 0.99 (0.90) | 0.65 (0.02) |
| `recurrent` | 0.99 (0.96) | 0.99 (0.96) | 0.89 (0.10) |
| `tabular` | 0.89 (0.86) | 0.90 (0.82) | 0.90 (0.84) |
| `random` | 0.30 (0.22) | 0.26 (0.14) | 0.28 (0.10) |

#### Rule A again: first 20-episode window 90% fed, median episode (lives / lives)

| Arm | D=2 | D=5 | D=10 |
| --- | --- | --- | --- |
| `live` | 17 (10/10) | 12 (10/10) | 25 (3/10) |
| `copy` | 8 (10/10) | 30 (8/10) | 241 (2/10) |
| `step` | 0 (10/10) | 0 (5/10) | 74 (4/10) |
| `lambda-zero` | 50 (7/10) | 24 (1/10) | none (0/10) |
| `yoked` | none (0/10) | none (0/10) | none (0/10) |
| `frozen` | 0 (10/10) | 0 (10/10) | 0 (4/10) |
| `blind` | 11 (10/10) | 9 (10/10) | 38 (6/10) |
| `recurrent` | 0 (10/10) | 0 (10/10) | 161 (9/10) |
| `tabular` | 2 (10/10) | 0 (10/10) | 2 (10/10) |
| `random` | none (0/10) | none (0/10) | none (0/10) |

#### Rule A again: wrong interactions per episode in the last 50, mean (minimum)

| Arm | D=2 | D=5 | D=10 |
| --- | --- | --- | --- |
| `live` | 0.50 (0.00) | 0.13 (0.00) | 1.22 (0.00) |
| `copy` | 0.92 (0.00) | 0.54 (0.00) | 2.20 (1.32) |
| `step` | 0.82 (0.04) | 0.50 (0.02) | 0.11 (0.04) |
| `lambda-zero` | 0.24 (0.00) | 0.83 (0.74) | 1.56 (1.22) |
| `yoked` | 0.68 (0.42) | 0.82 (0.66) | 1.67 (1.30) |
| `frozen` | 0.36 (0.00) | 0.58 (0.00) | 0.00 (0.00) |
| `blind` | 0.21 (0.00) | 0.23 (0.00) | 0.74 (0.00) |
| `recurrent` | 0.32 (0.00) | 0.21 (0.00) | 0.17 (0.02) |
| `tabular` | 0.20 (0.06) | 0.90 (0.46) | 1.74 (1.58) |
| `random` | 1.51 (1.32) | 3.00 (2.68) | 5.47 (5.14) |

#### The live arm: arousal and work, medians over lives

| Delay | aroused, whole life | aroused, second half of rule A | sweeps per routine moment | per aroused moment | learning sweeps | probe sweeps | memory reads | memory writes | routine ms (median, p90) | aroused ms (median, p90) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 0.167 | 0.037 | 22.5 | 21.3 | 44622 | 20128 | 47712 | 3944 | 0.66, 2.49 | 2.07, 13.35 |
| 5 | 0.171 | 0.026 | 29.0 | 22.0 | 49653 | 20160 | 47857 | 3895 | 0.64, 1.47 | 1.75, 6.92 |
| 10 | 0.861 | 0.807 | 23.8 | 22.3 | 149403 | 20128 | 62704 | 18980 | 0.64, 1.40 | 1.79, 3.94 |

#### Gates

```json
{
 "10": {
  "acquired": 0.2,
  "adapted": 0.3,
  "calm": 0.1,
  "crashed": 0,
  "frugal": 0.1,
  "lives": 10,
  "retained": 0.2
 },
 "2": {
  "acquired": 0.8,
  "adapted": 0.9,
  "calm": 0.7,
  "crashed": 0,
  "frugal": 0.7,
  "lives": 10,
  "retained": 0.9
 },
 "5": {
  "acquired": 0.8,
  "adapted": 0.9,
  "calm": 0.6,
  "crashed": 0,
  "frugal": 0.9,
  "lives": 10,
  "retained": 0.9
 },
 "passed": false,
 "pooled": {
  "acquired": 0.8,
  "adapted": 0.9,
  "calm": 0.65,
  "crashed": 0,
  "frugal": 0.8,
  "lives": 20,
  "retained": 0.9
 }
}
```


The failures of the third freeze stand with the first two. Two lives per delay end rule A below
the feeding bound and the calm bound is also missed. These readings do not isolate a
mechanism responsible for the misses. Return/reacquisition is positive under the historical
marginal criterion; the recurrent comparison needs the correction above, and the copy
variant did not improve this chamber.

## What runs

A creature walks a corridor once per trip: empty floor, a chest, a lamp, `D` levers and a
door, met in that order, 15 cells in all, so that every trip takes the same number of
moments whatever the delay. At every cell it passes or interacts. Under rule A the chest
holds the key; under rule B the lamp does. Interacting at the door with the key in the
pouch pays +1 and ends the trip. Taking the available key pays nothing; other interactions
at a chest, lamp or lever cost 0.25, including while already holding the key. Floor
interactions and a door interaction without the key pay zero. The door follows `L + 2`
cells after the chest or `L + 1` after the lamp, where `L` is the actual lever count, which
varies by up to two from trip to trip, varying the delay in event counts. This does not test
irregular physical time, clock input or cadence invariance. Intervening choices make the
last action alone an insufficient account of the earlier key-taking action. The creature
sees the kind of the cell it faces and, through the
pouch sense, whether it holds the key. One trip in twenty is cut short before the door at
a random cell, the key lost with it; such a trip ends with `done` clear, so the forecast
carries over into the next trip (a truncated bootstrap), while the door's end is terminal.
The outcome of a trip's last cell is delivered with the first observation of the next
trip, as `step` and `live` define it. The final action still has an undelivered outcome
when the run stops; the current receipt records it. A life is one stream without resets:
rule A for 500 trips, rule B for 500, then rule A again for 500. Nothing announces the change. The delays are
2, 5 and 10. The reward source moves; the goal and required reward rate stay fixed. This
is a contingency-reversal test, not a changed-goal or devaluation test.

The need of this body is 0.03 reward per moment, a gene of `ArousalConfig` added for this
chamber: at full feeding the income is 1/15 per moment, so a fed creature's need is met and
a starving one wants its whole need. Its selection and the reason for its law are recorded
[below](#how-the-operating-point-and-the-founders-were-selected).

In the current instrument every arm of a seed shares the corridor, cut and food-availability
schedules. Historical `/1` arms shared only the uncut layouts, as noted above:

| Arm | What it is |
| --- | --- |
| `live` | `Brain.compose(6, 2, modules=(32,))` at the protocol's operating point, with `ArousalConfig()` at its founders and the chamber's need, through `Brain.live` |
| `copy` | the live arm with the protocol’s declared efference-copy variant; the founder value zero is the live control |
| `recurrent` | an online Elman actor-critic over the same cell and pouch inputs, with one-moment gradients and eligibility traces; revision 2 corrects its update and clipping |
| `step` | the same brain and operating point without arousal, through `step`: it samples and learns at every moment (the simpler control) |
| `lambda-zero` | the `live` brain with eligibility decay `lam` at zero, removing direct eligibility credit to earlier actions; bootstrapped value learning remains |
| `yoked` | the `live` brain whose own door reward is banked and paid at a random cell of the next trip; it does not receive a paired `live` arm's rewards |
| `frozen` | the `live` brain after rule A, answering greedily and receiving no outcome, including the final rule-A outcome still pending at the switch |
| `blind` | a brain built without the pouch input; retained internal state remains available, and this arm does not isolate the working trace |
| `tabular` | epsilon-greedy Q(lambda) over (cell, pouch), the conventional online learner with the same information; its settings selected on the development seeds |
| `random` | uniform random actions |

The operating point of one continuing stream is the odour nursery's (working-trace
amplitude 0.3, consolidation 0.25, actor rate 0.1 with a bias rate of 0.01) with the
critic's rate at 5.0 in place of the composed 0.3, the eligibility decay at 0.95 in place
of 0.8 and the discount at 0.95 in place of 0.9; the composed values are the controls, and
the selection on the development seeds is recorded
[below](#how-the-operating-point-and-the-founders-were-selected).

Readings per rule, over the trips that reached the door: the share of trips that ended with
food, the share in which the key was taken, the wrong interactions per trip and the share
of door visits with the key at which the creature interacted, each over the last 50 trips;
the lag, the zero-based index among completed, non-cut trips of the first 20-trip window
with at least 90% fed (not a sustained-acquisition criterion); the number of trips
cut short; the greedy choice and the policy's probability of interacting, per cell and
pouch state, of a saved and reloaded copy every 25 trips; the probability of interacting
under the behaviour that acted and under the base policy, per cell and pouch state and
over the visits, read from the living brain; the share of aroused moments over the rule
and over its second half; and the work of the life: moments and settling sweeps per mode,
learning sweeps, probes, checkpoints, life and probe memory reads, memory writes, brains,
refused sweeps and the latency of a moment in each mode. Greedy probes use saved copies
with the inherited working trace and are reported separately from executed behavior.
Current reports include a probe at the actual phase end and door openings within the
last 50 completed trips. Historical `/1` endpoint probes were taken before trip 475 of
500, and its opening statistic covered the last 50 keyed door visits, potentially spanning
a longer period. Historical memory-read counts omitted both saved-copy probes and the
direct record-forecast recall. Current counts include both, with probe reads separate.
Completed routine forecasts preceding a refused action are counted separately in
`aborted_forecast_sweeps`; `refused_sweeps` counts the failing solve. These are
algorithmic work counts, not measurements of electrical energy. The conventional controls
also record action/probe calls, updates and parameter counts; recurrent actor and critic
clip counts are separate. Moment latency and total elapsed time include their compute.
These are not matched FLOP or energy budgets. Recurrent probes start from zero hidden state;
brain probes inherit saved context, so those probe tables are not matched state ablations.

## Historical gates, fixed before each confirmation attempt

Over the confirmation lives of the `live` arm at delays 2 and 5, at least 90% end each
rule with 90% of their last 50 trips fed, waste at most 1.5 wrong interactions per trip in
the last 50 trips of each rule, and spend no more than 35% of the second half of each rule
aroused. Delay 10 is reported without a gate. [protocol.json](protocol.json) holds the
gates, the seeds and every setting. It was committed and pushed before its confirmation
seeds were run. It is the chamber's second freeze; [the first](#the-first-freeze-2026-10-06)
gated frugality at 0.5 wrong interactions per trip and calm at 20%, and is recorded below
with its result. The frugality bound of 1.5 equals the uniform-random expectation at delay
2, not a strict improvement over it; the recorded random means below are 1.45 and 1.49.
It admits one retained habit per trip. The calm bound was relaxed to 35% after observing
arousal bouts following cut trips in the first freeze. A 5% cut rate does not derive a
35% arousal bound or imply that every cut contradicts a forecast. These revised criteria
are historical evaluation choices, not a pass of the original stricter contract. The
pooled gate does not require 90% success separately at each delay or that the same lives
pass every component. Both frozen attempts failed even this pooled test.

## Run and verify

```sh
python benchmarks/keydoor/key_door.py --arms live tabular random --seeds 0 --delays 2 --episodes 20 --workers 1 --out /tmp/keydoor.json.gz
python benchmarks/keydoor/key_door.py --verify /tmp/keydoor.json.gz --current
python benchmarks/keydoor/key_door.py --report /tmp/keydoor.json.gz
python -m pytest -q benchmarks/keydoor
```

The first command runs a short development check of the corrected instrument, not a
confirmation campaign. The third freeze contained ten arms, three delays and
ten confirmation seeds, 300 lives; the earlier two-rule design had 240 lives. `--arms`, `--seeds` and `--delays` select a part and make the current census ineligible
for confirmation.
`--genes`, `--point`,
`--cost`, `--food` and `--episodes` override the arousal genes, the operating point and the
world, and mark the receipt `frozen_protocol: false`; a `null` in `--point` leaves that
setting at its released default, and `--point '{"learner": {...}}'` sets the settling
learner. The receipt is a `cadence.Receipt` bound to the chamber's source and every module
of the library. `--verify` checks its canonical form and digest, source manifest integrity,
one row for every planned life in order, the ranges of the recorded shares, the visits
against the moments lived, the trips that reached the door and the cut trips against the
trips planned, the gates recomputed from the rows, the embedded protocol text against its
hash and any claim of frozen settings. `--current` also requires the source manifest and
the protocol hash of the files present. Historical receipts should be verified without
`--current`; their old source identities do not match the corrected instrument. The
verifier reports their historical limits explicitly, and `--report` verifies a receipt before
rendering it. Revision 2 additionally recomputes summaries from the stored trip outcomes
and moment sums, validates policy/probe tables, and checks the full-census joint audit.
Historical receipts contain summaries only. Neither format records every executed action
or independently proves that a run occurred.
`--report` prints the tables. `results/` keeps the receipts quoted here. The guards run
short lives of the `live` arm and its controls, the world's accounting through what the
learner is handed (the door terminal, a cut trip not), the checkpoint continuation inside
the delay with the preceding outcome pending, the independence of a life from its probes,
a refused answer charged and recorded, the gate arithmetic and the receipt's custody.

## Historical second-freeze results, 2026-10-06 (`key-door/1`)

Receipt: `results/confirmation-2026-10-06.json.gz`. Protocol SHA-256
`8cc9ed188c91a7f7098a978b2e5f963f43b8d5e46797d4be5144a66ee9b8124d`, frozen at commit
`b57f9fd`; seeds 800 to 809; cadence 0.75.0 with the `need` gene of this branch, NumPy 2.5.3,
Python 3.13.0, macOS arm64. All 240 lives completed and none refused an answer. **Three of
the four gates passed and the fourth did not, so the gates did not pass.** Of the 20 gated
lives of the `live` arm, 20 acquired rule A (fed 1.00 at delay 2 and 0.99 at delay 5 at its
end, median lags 9 and 66 trips), 19 re-adapted after the key moved (fed 0.97 and 1.00 at the
end of rule B, median lags 27 and 38 completed trips, and the historical conditional
door-opening statistic was 99%),
19 were frugal (0.31 and 0.13 wrong interactions per trip at the end of rule A, 0.45 and 0.24
at the end of rule B) and 17 were calm against the 18 required. The misses: one life at delay
2 kept touching the empty chest and the lamp while holding the key after the move, ended rule
B fed on 0.70 of its trips with 2.1 wrong interactions per trip and spent 69% of the late
moments aroused; one life at delay 2 spent 54% of the late half of rule A aroused while fed on
every trip without a wrong interaction; one life at delay 5 acquired at trip 316 and
re-adapted at trip 373 and was still searching in the second half of each rule (32% and 46%).
At delay 10, ungated, 5 of 10 lives acquired rule A and 8 of 10 ended rule B fed (0.89 at its
end, median lag 172 trips; the tabular learner 0.88).

The historical controls, with the unmatched-cut limitation above: `step`, learning at every moment, ended rule B fed at 0.97
and 0.68 while aroused at every moment and wasting 1.8 and 0.7 interactions per trip;
`lambda-zero` at 0.68 and 0.33 with direct eligibility credit restricted to the last action;
`yoked`, with its own earned door rewards retimed, at 0.28 and 0.07;
`frozen` at 0.00 (no adaptation without outcomes); `blind`, without the pouch sense,
at 1.00 and 1.00 with no wrong interaction at either delay. This does not demonstrate
trace-mediated key memory: a current-cell policy can interact only at the rewarded
container and the door. No trace lesion or transplant isolates the mechanism here; `tabular` at 0.90
and 0.92, bounded by its epsilon; `random` at 0.22. The `live` arm's routine moment took
0.71 to 0.79 ms (median) against 2.0 to 2.2 ms aroused, at 25 to 31 settling sweeps per
routine moment and 22 to 24 per aroused moment. These within-life laptop measurements
show lower median latency during routine, which omits learning phases. They do not compare
matched states, establish reduced settling work or measure electrical energy. It spent
13% and 21% of its moments aroused over the whole life and
5% and 4% over the second half of rule A, wrote 1,870 and 2,943 records to memory and spent
12,800 sweeps per life on the probes of saved copies. The historical ledger omits probe
and direct record-forecast memory reads, and the final outcome remains undelivered. Retiming an arm's own
rewards changes its later behavior and reward count; this control cannot by itself isolate
TD credit from altered reward timing and state correlations.

### Declared variants on the same seeds

`results/variant-scarcity-2026-10-06.json.gz` (`--food 0.5`, the door paying one time in two
with the key): fed shares were 0.25 and 0.22 at the end of rule A, and 0.35 and 0.27 at
the end of rule B. The original 90%-fed acquisition gate is unsuitable here: even a
policy that always obtains the key and opens the door has expected feeding of 50%.
Failure of that gate does not mean nothing was learned. Key-taking shares were 0.69 and
0.60 in A and 0.74 and 0.65 in B; the lives remained substantially aroused (83% to 91%
of moments across the delay/rule groups). Ideal uncut gross income is about 0.036 per
moment, above the need of 0.03; cuts and costs reduce it. The summaries alone do not
establish the cause of persistent arousal or a saturation latch in every life. No new
scarcity success gate is fitted to these results.
`results/variant-composed-critic-2026-10-06.json.gz` (critic rate 0.3, eligibility decay 0.8,
discount 0.9, all else the frozen point): 13 of 20 lives acquired rule A and 11 re-adapted
(at delay 5, 4 and 3 of 10), against 20 and 19 for the frozen point.

### The tables

The tables retain the historical `/1` receipt values; corrected `/2` measurement semantics
do not retroactively change them. Lag units are completed, non-cut trips. Door openings,
probes and work retain the historical limitations stated above.

### Rule A, chest holds the key: trips fed in the last 50, mean (minimum)

| Arm | D=2 | D=5 | D=10 |
| --- | --- | --- | --- |
| `live` | 1.00 (1.00) | 0.99 (0.92) | 0.69 (0.12) |
| `step` | 0.94 (0.86) | 0.90 (0.78) | 0.57 (0.00) |
| `lambda-zero` | 0.82 (0.04) | 0.51 (0.00) | 0.12 (0.00) |
| `yoked` | 0.35 (0.14) | 0.14 (0.02) | 0.15 (0.02) |
| `frozen` | 1.00 (1.00) | 0.99 (0.92) | 0.69 (0.12) |
| `blind` | 1.00 (1.00) | 1.00 (1.00) | 0.57 (0.04) |
| `tabular` | 0.68 (0.00) | 0.89 (0.84) | 0.89 (0.84) |
| `random` | 0.25 (0.16) | 0.26 (0.16) | 0.23 (0.14) |

### Rule B, lamp holds the key: trips fed in the last 50, mean (minimum)

| Arm | D=2 | D=5 | D=10 |
| --- | --- | --- | --- |
| `live` | 0.97 (0.70) | 1.00 (0.98) | 0.89 (0.02) |
| `step` | 0.97 (0.90) | 0.68 (0.02) | 0.43 (0.00) |
| `lambda-zero` | 0.68 (0.02) | 0.33 (0.00) | 0.13 (0.00) |
| `yoked` | 0.28 (0.06) | 0.07 (0.02) | 0.13 (0.00) |
| `frozen` | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) |
| `blind` | 1.00 (1.00) | 1.00 (1.00) | 0.70 (0.02) |
| `tabular` | 0.90 (0.84) | 0.92 (0.82) | 0.88 (0.78) |
| `random` | 0.22 (0.10) | 0.22 (0.18) | 0.28 (0.12) |

### Rule A: key taken in the last 50 trips, mean (minimum)

| Arm | D=2 | D=5 | D=10 |
| --- | --- | --- | --- |
| `live` | 1.00 (1.00) | 1.00 (0.98) | 0.76 (0.34) |
| `step` | 0.95 (0.90) | 0.92 (0.80) | 0.59 (0.04) |
| `lambda-zero` | 0.85 (0.22) | 0.59 (0.10) | 0.24 (0.06) |
| `yoked` | 0.52 (0.28) | 0.38 (0.18) | 0.35 (0.18) |
| `frozen` | 1.00 (1.00) | 1.00 (0.98) | 0.76 (0.34) |
| `blind` | 1.00 (1.00) | 1.00 (1.00) | 0.63 (0.22) |
| `tabular` | 0.75 (0.04) | 0.95 (0.90) | 0.93 (0.90) |
| `random` | 0.49 (0.38) | 0.50 (0.32) | 0.50 (0.44) |

### Rule B: key taken in the last 50 trips, mean (minimum)

| Arm | D=2 | D=5 | D=10 |
| --- | --- | --- | --- |
| `live` | 0.98 (0.80) | 1.00 (1.00) | 0.90 (0.14) |
| `step` | 0.98 (0.94) | 0.69 (0.04) | 0.44 (0.00) |
| `lambda-zero` | 0.72 (0.14) | 0.41 (0.12) | 0.25 (0.08) |
| `yoked` | 0.49 (0.18) | 0.23 (0.14) | 0.28 (0.12) |
| `frozen` | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) |
| `blind` | 1.00 (1.00) | 1.00 (1.00) | 0.75 (0.10) |
| `tabular` | 0.94 (0.86) | 0.97 (0.94) | 0.94 (0.88) |
| `random` | 0.49 (0.42) | 0.48 (0.36) | 0.49 (0.38) |

### Rule A: wrong interactions per trip in the last 50, mean (minimum)

| Arm | D=2 | D=5 | D=10 |
| --- | --- | --- | --- |
| `live` | 0.31 (0.00) | 0.13 (0.00) | 1.53 (0.00) |
| `step` | 0.82 (0.02) | 0.55 (0.06) | 0.31 (0.12) |
| `lambda-zero` | 0.53 (0.00) | 0.62 (0.00) | 1.39 (0.00) |
| `yoked` | 0.78 (0.02) | 1.53 (0.84) | 2.96 (1.84) |
| `frozen` | 0.31 (0.00) | 0.13 (0.00) | 1.53 (0.00) |
| `blind` | 0.00 (0.00) | 0.00 (0.00) | 1.62 (0.00) |
| `tabular` | 0.15 (0.04) | 0.49 (0.28) | 1.62 (1.10) |
| `random` | 1.45 (1.20) | 2.98 (2.68) | 5.47 (5.14) |

### Rule B: wrong interactions per trip in the last 50, mean (minimum)

| Arm | D=2 | D=5 | D=10 |
| --- | --- | --- | --- |
| `live` | 0.45 (0.00) | 0.24 (0.00) | 0.43 (0.00) |
| `step` | 1.78 (0.92) | 0.67 (0.10) | 0.50 (0.06) |
| `lambda-zero` | 0.39 (0.00) | 0.70 (0.00) | 1.37 (0.00) |
| `yoked` | 0.88 (0.00) | 1.09 (0.68) | 2.26 (1.46) |
| `frozen` | 1.00 (1.00) | 1.00 (1.00) | 0.77 (0.00) |
| `blind` | 0.00 (0.00) | 0.00 (0.00) | 0.87 (0.00) |
| `tabular` | 0.15 (0.06) | 0.53 (0.34) | 1.66 (1.06) |
| `random` | 1.49 (1.30) | 2.91 (2.36) | 5.36 (4.74) |

### Rule A: first 20-completed-trip window 90% fed, median zero-based index (lives / lives)

| Arm | D=2 | D=5 | D=10 |
| --- | --- | --- | --- |
| `live` | 9 (10/10) | 65 (10/10) | 303 (8/10) |
| `step` | 122 (10/10) | 188 (10/10) | 250 (3/10) |
| `lambda-zero` | 14 (8/10) | 6 (6/10) | 9 (2/10) |
| `yoked` | 49 (4/10) | 132 (2/10) | 50 (2/10) |
| `frozen` | 9 (10/10) | 65 (10/10) | 303 (8/10) |
| `blind` | 14 (10/10) | 113 (10/10) | 56 (6/10) |
| `tabular` | 20 (8/10) | 17 (10/10) | 33 (10/10) |
| `random` | none (0/10) | none (0/10) | none (0/10) |

### Rule B: first 20-completed-trip window 90% fed, median zero-based index (lives / lives)

| Arm | D=2 | D=5 | D=10 |
| --- | --- | --- | --- |
| `live` | 26 (10/10) | 38 (10/10) | 172 (9/10) |
| `step` | 23 (10/10) | 122 (8/10) | 140 (6/10) |
| `lambda-zero` | 29 (7/10) | 15 (3/10) | 116 (1/10) |
| `yoked` | 27 (2/10) | none (0/10) | 464 (1/10) |
| `frozen` | none (0/10) | none (0/10) | none (0/10) |
| `blind` | 13 (10/10) | 32 (10/10) | 311 (6/10) |
| `tabular` | 29 (10/10) | 9 (10/10) | 32 (10/10) |
| `random` | none (0/10) | none (0/10) | none (0/10) |

### Rule A behaviour: probability of taking at the chest without the key, median (minimum)

| Arm | D=2 | D=5 | D=10 |
| --- | --- | --- | --- |
| `live` | 0.965 (0.659) | 0.863 (0.558) | 0.555 (0.407) |
| `step` | 0.844 (0.666) | 0.793 (0.592) | 0.464 (0.132) |
| `lambda-zero` | 0.972 (0.241) | 0.728 (0.186) | 0.214 (0.172) |
| `yoked` | 0.486 (0.408) | 0.454 (0.338) | 0.428 (0.276) |
| `frozen` | 0.965 (0.659) | 0.863 (0.558) | 0.555 (0.407) |
| `blind` | 0.971 (0.544) | 0.849 (0.518) | 0.478 (0.227) |
| `tabular` | 0.939 (0.056) | 0.938 (0.691) | 0.889 (0.291) |
| `random` | 0.500 (0.500) | 0.500 (0.500) | 0.500 (0.500) |

### Rule B behaviour: probability of taking at the lamp without the key, median (minimum)

| Arm | D=2 | D=5 | D=10 |
| --- | --- | --- | --- |
| `live` | 0.956 (0.747) | 0.920 (0.477) | 0.726 (0.217) |
| `step` | 0.936 (0.731) | 0.598 (0.028) | 0.267 (0.015) |
| `lambda-zero` | 0.899 (0.154) | 0.176 (0.149) | 0.144 (0.141) |
| `yoked` | 0.424 (0.349) | 0.224 (0.152) | 0.280 (0.155) |
| `frozen` | 0.000 (0.000) | 0.000 (0.000) | 0.000 (0.000) |
| `blind` | 0.978 (0.829) | 0.950 (0.689) | 0.466 (0.156) |
| `tabular` | 0.902 (0.757) | 0.927 (0.870) | 0.911 (0.857) |
| `random` | 0.500 (0.500) | 0.500 (0.500) | 0.500 (0.500) |

### Rule B behaviour: probability of interacting at a lever with the key, median (minimum)

| Arm | D=2 | D=5 | D=10 |
| --- | --- | --- | --- |
| `live` | 0.045 (0.006) | 0.024 (0.006) | 0.063 (0.012) |
| `step` | 0.315 (0.035) | 0.041 (0.020) | 0.019 (0.011) |
| `lambda-zero` | 0.135 (0.007) | 0.147 (0.005) | 0.141 (0.034) |
| `yoked` | 0.365 (0.022) | 0.226 (0.149) | 0.267 (0.148) |
| `frozen` | none | none | none |
| `blind` | 0.007 (0.002) | 0.011 (0.001) | 0.151 (0.006) |
| `tabular` | 0.051 (0.050) | 0.092 (0.080) | 0.148 (0.137) |
| `random` | 0.500 (0.500) | 0.500 (0.500) | 0.500 (0.500) |

### Historical live-arm arousal and recorded work, medians over lives

| Delay | aroused, whole life | aroused, second half of rule A | sweeps per routine moment | per aroused moment | learning sweeps | probe sweeps | memory reads | memory writes | routine ms (median, p90) | aroused ms (median, p90) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 0.131 | 0.053 | 25.3 | 23.9 | 19287 | 12800 | 15449 | 1870 | 0.71, 2.42 | 2.01, 8.40 |
| 5 | 0.210 | 0.042 | 31.0 | 22.0 | 33100 | 12800 | 16556 | 2943 | 0.79, 2.43 | 2.16, 10.27 |
| 10 | 0.461 | 0.494 | 26.4 | 23.1 | 67488 | 12816 | 20110 | 6444 | 0.68, 2.03 | 1.94, 6.00 |

### Gates

```json
{
 "10": {
  "acquired": 0.5,
  "adapted": 0.8,
  "calm": 0.4,
  "crashed": 0,
  "frugal": 0.5,
  "lives": 10
 },
 "2": {
  "acquired": 1.0,
  "adapted": 0.9,
  "calm": 0.8,
  "crashed": 0,
  "frugal": 0.9,
  "lives": 10
 },
 "5": {
  "acquired": 1.0,
  "adapted": 1.0,
  "calm": 0.9,
  "crashed": 0,
  "frugal": 1.0,
  "lives": 10
 },
 "passed": false,
 "pooled": {
  "acquired": 1.0,
  "adapted": 0.95,
  "calm": 0.85,
  "crashed": 0,
  "frugal": 0.95,
  "lives": 20
 }
}
```

## The first freeze, 2026-10-06

Receipts: `results/freeze1-confirmation-2026-10-06.json.gz` and the two variants beside it.
Protocol SHA-256 `2f214aaebff44070e1c0428fec3159005cef7f2ec13cb949dddf5b15b7e170c0`, frozen at
commit `35fcb14`; seeds 700 to 709; the same world, operating point and need as the second
freeze, with the gates at 0.5 wrong interactions per trip and 20% late arousal. **Its gates
did not pass.** Of the 20 gated lives of the `live` arm, 18 ended rule A fed and all 20
ended rule B fed (fed 1.00 at both delays, median lags of 28 and 34 trips after the key
moved, with the key taken on every trip); 11 were frugal and 14 calm by the first gates.
The misses were of three kinds: seven lives kept one extra interaction per trip while fed,
at the empty chest under rule B or at the lamp while already holding the key under rule A
(wrong 1.0 to 1.2 with fed 1.00; one life 2.6); six spent 22% to 36% of a late half aroused
while fed and frugal, in bouts that follow the cut trips; and one life at delay 5 never
acquired rule A (fed 0.24, aroused 87% of the late half) and then re-adapted to 1.00 under
rule B. At delay 10, 4 of 10 acquired and 2 of 10 re-adapted. The controls: `step` ended
rule B fed at 0.98 and 0.77 while aroused at every moment and wasting 0.8 to 1.5
interactions per trip; `lambda-zero` 0.51 and 0.30; `frozen` 0.00 (no adaptation without
outcomes); `blind`, without the pouch sense, 0.98 and 0.91 (this does not isolate trace
memory); `tabular` 0.90 and 0.92, and 0.91 at delay 10 where the brain reached 0.48;
`random` 0.25. The `yoked` arm crashed in 28 of 30 lives on a trip cut to one cell, where
its random payment cell had no range (`integers(0)`): a fault of that control alone,
repaired in the second freeze. The scarcity variant (`--food 0.5`, the door paying one time
in two) recorded fed shares of 0.14 to 0.36 and arousal of 88% to 92%. Its unchanged
90%-fed gate is unsuitable for a world that pays a successful door opening with
probability 0.5; these failures do not establish absence of acquisition. The composed-critic variant (critic rate 0.3,
eligibility decay 0.8, discount 0.9, all else the frozen point) acquired 11 of 20 and
re-adapted 10 of 20, against 18 and 20 for the frozen point. The gates of the second
freeze were revised on this record as stated above; the point and the world were not.

## How the operating point and the founders were selected

The following are source-specific development observations and proposed explanations.
They are not independent causal demonstrations or guarantees for the corrected source.

Development used seeds 0 to 3 at every delay; the confirmation seeds were first run on the
frozen protocol. The odour nursery's operating point (working-trace amplitude 0.3,
consolidation 0.25, actor rate 0.1 with a bias rate of 0.01) was the starting point, and
the chamber was built up from a short corridor to the one above.

- **A need, and its unit.** With the arousal founders of 0.75.0 a creature whose food
  stops is roused only by the shortfall of its recent reward below its long-run reward,
  and a newborn that has never been paid is content with nothing: at delay 5 no
  development life acquired the task. A need, the reward per moment the body requires,
  rouses both. Measured in the law's unit, the spread of the outcomes the brain has
  learned from, the need did not work on this corridor: a reward of +1 once in 14 moments
  has a mean of 1/14 and a spread near 1/sqrt(14), so a starving creature's want stayed
  near 0.2 whatever the need, at the edge of the threshold, and its exploration heat
  barely rose; on the 24 development lives of the first grid 1 of 24 re-adapted after the
  key moved. The need's want is therefore the share of the need the recent reward leaves
  unmet, 1 for a creature that is never paid and 0 for one whose income covers it. A need
  of zero removes it and leaves the law of 0.75.0 unchanged. With the need at 0.03 and an
  eligibility decay of 0.9, every development life at delay 2 acquired the task and
  re-adapted after the key moved; delay 5 did not.
- **Corridor length.** In the first corridor the trip grew with the delay (`D + 3`
  cells), so the achievable income per moment fell with the delay and no single need
  suited every rung. Every trip now has 14 cells, with floor cells before the chest making
  up the difference, so the income at full feeding is the same at every delay.
- **The cost of a wrong interaction.** At a cost of 0.1 the creatures that acquired the
  task interacted at every cell, levers included, and still fed: the frugal policy pays
  little more than the lazy one at delay 2. At 0.25 the lazy policy loses, and the
  creatures that learn the task stop touching the levers. The cost of 0.25 stays.
- **The latch under punished exploration.** At delay 5 the development lives that failed
  did so in one way, read from the living brain every 100 trips: within 200 to 300 trips
  the two motor units settled to the same activation for every cell and pouch state (a
  cosine of 1.000 between all ten situations), the pass unit near 0.95, the interact unit
  below rest, and the policy answered pass with a base probability of 0.005 everywhere.
  These readings suggest saturation limited the nudge in the observed runs; they do not
  prove that no later reward or admissible experience could change the state.
  The synapses stayed near initialization (mean efficacy 1.38 at birth, 1.41 after 300
  trips) and the motor bias reached 0.21, so the efficacy cap of 8 could not bite. The
  calm policy of a fed creature sits on motor activations near rest, with margins of a
  hundredth, so when its food stops and its arousal samples at three times the
  temperature, the behaviour is close to uniform, the levers punish about half the cells
  of every trip, and the flood of negative dopamine drives the pass unit to its rail.
  Dopamine centring made every nonzero outcome a unit signal and hastened the collapse;
  a sharper learner temperature of 0.1 or 0.05 deepened it; a heat of 0.5, an actor rate
  of 0.03, a trace amplitude of 0, a bias rate of 0, 64 modules and an efficacy cap of 2
  changed nothing. The lives that acquired rule A at delay 5 under these settings never
  re-adapted after the key moved; the same mechanism is a hypothesis for those failures.
- **The critic carries delayed credit.** The brains that acquired rule A at delay 5 had a
  flat critic: a value of 0.1 to 0.2 at every cell, 0.2 at the door with the key where +1
  follows with certainty, and the same value with and without the key. The composed
  critic learns at a rate of 0.3 divided by one plus the energy of its eligibility trace,
  which over a 14-cell trip is about 27, on a code whose activations average 0.03; it
  learns its bias and little else. Taking the key then produces no value jump, and the
  only credit for the chest is the actor's own eligibility, (gamma * lam)^k with k the
  cells to the door: 0.43 at delay 2 and 0.28 at delay 5. This is a candidate explanation
  for the observed delay difference, not an isolated causal result. Raising the critic's rate makes the key's value appear (about
  +0.4 for holding it at the end of rule A) and the lives at delay 5 acquire and
  re-adapt; the rate is selected below with the composed 0.3 as the control. An
  unnormalized critic at 0.05 did the same for one seed and was slower for the other.
- **The other rail.** At a critic rate of 10 the values run high (above 1 at the door
  with the key, 0.4 on the empty floor) and after the key moves the development lives
  drift to the opposite latch: by trip 250 of rule B they interact at every cell with a
  behaviour probability of 0.85, five wrong interactions per trip, fed on two trips in
  three with a net income below zero, so the need is never met, the want stays at 1 and
  the creature never returns to routine; 800 trips did not end it. A rate of 30 reached
  that state under rule A. The rate is therefore chosen between the flat critic and the
  running one, on the readings of all three gates rather than on the fed share alone.
- **The point.** On seeds 0 to 3 at the three delays, critic rates of 3, 5 and 10 with an
  eligibility decay of 0.9 all acquired rule A at delays 2 and 5 in every life; after the
  key moved at delay 5, 3, 2 and 4 of the 4 lives re-adapted, the last with one wrong
  interaction per trip and a third of the late moments aroused. Rate 5 with the eligibility
  decay at 0.95 and the discount at 0.95, which carry the door's credit further back
  ((gamma * lam)^6 of 0.52 against 0.28), re-adapted 3 of 4 with 0.3 wrong interactions
  per trip and 12% of the late moments aroused. On seeds 0 to 7 at delays 2 and 5, with
  receipts, that point passed every gate reading in 6 of 8 lives at each delay (fed at
  the end of rule A 15 of 16, of rule B 15 of 16, frugal 13 of 16, calm 13 of 16); rate 5
  at the composed decay and discount passed 5 of 8 and 4 of 8, rate 3 there 3 of 8 and
  2 of 8. The misses of the frozen point: two lives at
  delay 2 kept interacting at the lamp while holding the key, a habit from acquisition
  that costs a quarter per trip and that a fed and calm creature does not unlearn; one
  life at delay 5 latched after the key moved; one life at delay 5 spent 21% of the late
  moments of rule A aroused. The frozen point is rate 5, decay 0.95, discount 0.95, and
  these development comparisons used the first gates; the later second freeze relaxed
  frugality and calm as disclosed above. Delay 10 is left ungated: 0 to 2 of 4 development
  lives acquired it under any point tried, while the tabular learner acquires it.
- **The tabular learner.** Alpha in 0.1, 0.2 and 0.5, epsilon in 0.05, 0.1 and 0.2 and
  lambda in 0.8 and 0.9 were run on the development seeds at every delay, with gamma 0.9.
  Epsilon 0.05 left lives that never found the pair of interactions and epsilon 0.2 wasted
  one to three interactions per trip; at epsilon 0.1 every life ended both rules fed at
  0.74 or more, and alpha 0.5 with lambda 0.8 had the highest shares (0.89 to 0.93 under
  rule A, 0.89 to 0.92 under rule B, at the three delays) with lags of 7 to 95 trips. Its
  epsilon bounds it: one random pass at the chest or the door in twenty costs the trip.
  It solves delay 10 within the 500 trips; the brain's gates stop at delay 5.
