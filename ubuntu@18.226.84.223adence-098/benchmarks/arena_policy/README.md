# Actual policy credit and bounded motor exploration

Issues [159](https://github.com/muellerberndt/cadence/issues/159) and
[160](https://github.com/muellerberndt/cadence/issues/160), development evidence,
2026-10-08. This instrument uses the real
[robot arena](https://github.com/muellerberndt/cadence-robot-arena), its unchanged
Tumbler body, observations, reward, and founder genes. It measures a continuing
reward life followed by private greedy behavior; there are no teachers or
external answer-producing components.

## The two changes and their controls

The actor previously sampled at an exploration temperature but computed the
local eligibility at the base temperature. When the critic's mean advantage is
nonzero, the difference between those distributions contributes an update even
without action-specific evidence. The repair uses the actual sampling
temperatures in the existing free/nudged local contrast. The common scale is
`c = min(base_temperature, min(actual_slot_temperatures))`; each slot's nudge
mask is `c / actual_slot_temperature`. Thus the contrast estimates the same
`c * gradient(log joint_policy)` across all slots with bounded nudge gains,
preserving the existing base-temperature law.
[Independent tests](../../tests/test_behavior_credit.py) check sensory-synapse
and motor-bias derivatives and zero expected score for action-independent reward
under scalar and unequal-slot temperatures. Finite phases retain their existing
approximation limits; this is not proof that every learning failure had this cause.

The second change confines the extra heat of an aroused multi-slot action to one
uniformly selected motor slot. All other slots still sample their base policies.
Every slot participates in the same equilibrium and receives credit for its actual
conditional sampling distribution. Single-slot exploration and zero heat keep
their existing sampling laws; no new gene or changed learning rate is introduced.

The three native arms are the income-repaired source without either policy
change (`control`), corrected actual-policy credit alone (`credit`), and credit
plus one heated slot (`combined`). All use the arena's existing founder recipe,
including actor eta 0.03, base temperature 0.5, and sensory projection scale 4.
Those are application genes, not new library defaults.

## Four-thousand-moment development screen

Founders 11, 12, and 13 use training world seeds 7, 8, and 9. After 4,000
moments, a saved copy is evaluated greedily for 1,000 moments with learning
disabled. Evaluation seeds are 19, 19, and 20. Uniform-random controls use the
same body and evaluation seed. Every outcome below remains in the record.

| Founder | Uniform random progress (m) | Newborn greedy progress (m) | Control greedy progress (m) | Credit-only greedy progress (m) | Combined greedy progress (m) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 11 | 0.711 | -5.117 | 0.264 | 11.719 | 0.000 |
| 12 | -1.246 | -0.162 | -7.911 | 3.717 | 7.074 |
| 13 | -2.110 | -0.551 | 0.467 | 6.039 | 14.062 |

The [frozen newborn comparator](frozen-newborn-2026-10-08.json) uses the same
founders, combined runtime, worlds and 1,000-moment evaluation lengths without
training or checkpoint loading. It deals no damage, destroys no dummies, and
performs no learning sweeps. These before-training controls were measured after
the trained comparisons; they are developmental evidence, not a preregistered
confirmation.

The credit correction improves frozen greedy progress against both controls in
all three development founders. The combined change improves it in two;
founder 11 remains stationary. It is therefore incorrect to claim that every
combined brain acquired steering after this short life. The combined founder
12 destroys one dummy and deals 115.07 damage, but takes 158.69 damage; progress
does not establish good combat tactics or safety from the ring. There are no
refused actions in these measurements.

The probability-range diagnostic does not monotonically increase: credit-only
sensitivities are 0.085, 0.060, and 0.117, against control 0.110, 0.063, and
0.128. Behavior is the acceptance measurement; simply increasing numerical
policy variation would not repair the reported limitation.

Private cue ablations zero only the first four sensory coordinates (the target's
bearing/proximity channels); body, ring and weapon senses remain. They replay the
same saved brain without learning at the same evaluation seed:

| Founder | Credit intact / no target channels (m) | Combined intact / no target channels (m) |
| --- | ---: | ---: |
| 11 | 11.719 / 0.056 | 0.000 / 0.000 |
| 12 | 3.717 / 0.000 | 7.074 / 0.000 |
| 13 | 6.039 / -0.151 | 14.062 / -0.089 |

These ablations support target-sensory participation in the positive greedy
results. They do not establish held-out robot bodies, moving-opponent tactics,
general transfer, or independent confirmation.

## Reported eighty-thousand-moment horizon

A fresh matched pair uses founder 11, training world 7, the same genes, and
80,000 moments. There is no parameter search or checkpoint selection. Every
10,000 moments the existing nursery records a private fingerprint. The final
saved brain receives the same 1,000-moment frozen evaluation at world 19:

| Policy | Progress (m) | Kills | Damage dealt / taken | Progress without target channels (m) |
| --- | ---: | ---: | ---: | ---: |
| Uniform random | 0.711 | 0 | 0 / 0 | — |
| Control after 80,000 moments | 17.463 | 3 | 163.79 / 41.26 | 3.273 |
| Combined after 80,000 moments | 13.285 | 2 | 100.42 / 53.59 | 0.821 |

Both sources retain target-sensitive greedy behavior above random at this
horizon. Both give four distinct answers across the eight situations, with
sensitivities 0.322 and 0.123 respectively. The combined source is weaker than
the control on this final fighting evaluation; the result supports preservation
of sensory behavior, not improved long-horizon fighting. This fresh control does
not reproduce the report's observation blindness. The combined source trains
with 18 dummy kills versus 8, but training experience is different and this does
not replace the frozen comparison. All training and evaluation actions succeed
without refusal.

## Existing acquired lives

The separate [positive-need continuation receipt](continuation-acquired-positive-need-2026-10-08.json)
and [source audit](continuation-acquired-positive-need-2026-10-08.sources.json)
continue the same six previously acquired arena brains in one 600-moment royale,
with their original needs and owed outcomes, using seed 77. Fighters that die
issue fewer than 600 actions. Compared with the
income correction alone, the combined policy changes reduce closing from
40.67% to 37.28% and damage dealt from 318.4 to 258.2. Uniform random closes
25.02% and deals no damage. Arousal falls from 78.93% to 75.90%, and learning
sweeps from 27,493 to 24,313, with no refusals. These inherited lives retain
above-random behavior, but this comparison does not show improved fighting.

## Preserved mixed diagnostic

A preceding synthetic two-cue, two-motor continuing reward screen used 4,000
outcomes, three founders, and the arena-like actor/trace rates. Old versus
corrected scalar-hot credit gave cold graph recall `[1.0, 1.0, 0.5]` versus
`[0.75, 0.75, 1.0]`: equal mean 0.833. The corresponding sampled success did not
uniformly improve. This is a mixed result, not an acquisition win. Its cold
`predict` readings omit the working trace and are distinct from the native
arena's private continuing greedy evaluation. The exact original producer and
all six rows are retained. It must not be replaced by the positive arena table.

## Source custody and reproduction

The gzip files in [results](results/) preserve every raw JSON/JSONL byte;
[manifest.json](results/manifest.json) lists compressed and uncompressed hashes.
Checkpoints stay in the arena's ignored `runs/` directory, outside the repository.
The initial founder-11 control/credit receipts have less source metadata: their
invocation pins were `bf90fa4` and `c3a1f67`, respectively, but no separate original
producer file or complete source-hash sidecar was recorded. Their receipts are
left unchanged. The seven other acquisition/evaluation records carry complete
Cadence and arena Python source hashes, checked before and after each comparison,
plus source commits. The six older separate ablations bind Cadence source only;
they have no independent arena source-hash sidecar. The synthetic rows also lack
an explicit runtime source pin. The combined founder-11 receipt has full runtime
hashes, but its exact original producer was not retained. Available original
producers are retained as text under [historical-producers](historical-producers/).
The combined additional source `ee5cc94` is runtime-identical to `cc71fe4`.
Both 80,000-moment receipts bind the complete Cadence and arena sources, the
saved checkpoint, and the exact producer. That original long-run producer is
retained as `historical-producers/compare-long-original.py.txt`; the maintained
`compare.py` only adds lint cleanup and clearer limitation text afterward.
The [source-equivalence map](source-equivalence.json) checks equality of the entire
committed `src/cadence` tree. Reproduction from the published parent history can
use `50c1c97` or `c3a1f67` for scalar credit (measurement `7e308f6`/`01a0400`),
and `cc71fe4` for the combined law (measurement `6fae103`/`ee5cc94`), without
depending on the local measurement branches. The source hashes remain the
authoritative identity of the measured runtime; documentation/tests can differ.

For a new source-bound run, use the arena environment and declare both source
checkouts explicitly. Use separate output folders for repeated attempts:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /path/to/arena/.venv/bin/python \
  benchmarks/arena_policy/compare.py \
  --cadence /path/to/cadence --arena /path/to/cadence-robot-arena \
  --output /path/to/local-runs/attempt-01 --arm combined --founders 11 12 13
```

The report's 80,000-moment horizon uses `--moments 80000 --founders 11
--probe-every 10000`, retaining the same genes and matched baseline. The runner
then records the full eight-situation fingerprint, greedy and uniform-random
world evaluations, and the target bearing/proximity ablation. The separate
[frozen_newborn.py](frozen_newborn.py) producer reproduces the untrained control.

`arena.run_nursery` leaves its final world reward undelivered. These receipts do
not claim a fully delivered final outcome or resumed training. Greedy evaluations
are private read-only lives. Identical initial world seeds do not force identical
later dummy relocation schedules when actions destroy dummies at different times.
Intermediate probe costs enter elapsed time, but their settling sweeps are not
included in the nursery's sweep counters. The final standalone fingerprint is
outside the reported nursery elapsed time and also lacks a settling-work count;
these receipts are not a complete resource ledger. These are development comparisons,
not a fresh frozen confirmation or a general robot-fighting capability claim.
