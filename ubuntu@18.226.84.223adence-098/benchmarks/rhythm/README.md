# Steady-rhythm chamber

This bounded instrument asks whether one continuing `Brain.compose` life can alternate
two learned actions, A, B, A, B, while every observation it receives is identical.
It is the continuing-action chamber of
[issue 116](https://github.com/muellerberndt/cadence/issues/116), roadmap row 06 of
[issue 109](https://github.com/muellerberndt/cadence/issues/109). It measures one
declared System 1 recipe and its compose-default control against lesioned forks of
the same checkpoint, a matched flip-flop control and a uniform-random baseline. It
establishes no default, no mechanism and no general timing capability. Read
[the protocol](RHYTHM_PROTOCOL.md), the [world-model guide](../../docs/start.md#one-continuing-equilibrium-brain)
and [numerical contracts](../../docs/contracts.md) before interpreting results.

## What runs

One brain with four continuing rows is taught over 24 bouts. A bout opens with one
cue event (two rows cued A, two cued B, by a frozen permutation) and continues with
eleven identical drive events. The label at the cue event is the cued action; at
every later event it is the opposite of the row's own last executed action. Lessons
go through `Learner.step(brain.stimulus(x), labels)` with the live trace in the
stimulus, followed by a free greedy `act`. The `every` arm teaches before each act;
the `mismatch` arm acts first and teaches only the rows that were wrong. The working
trace is the only history the brain carries; `Learner.step` leaves it untouched and
only the free act advances it.

After teaching, a final bout runs its cue and four lead events, then saves the probe
checkpoint between two actions. The live life and eight forks of that checkpoint run
the same 64 drive events: intact, restored, erased trace, shuffled trace (transplanted
from a row with the other cue), reset, static (trace erased before every act),
static_cold (`reset` before every act), the flip-flop control and frozen random
actions. Disturbance forks run 8 drive events, then a pause of 1, 2 or 4 zero-drive
events or one distractor event, then 32 drive events. Paced forks run 32 events at a
declared cadence of one event per 100 ms, at 50 ms and 200 ms, with an extra event
in one slot, with a skipped slot, and at 100 ms beside one spinning process per
logical CPU; unpaced forks vary the free-step budget and the residual tolerance. Every
variant's per-event actions are compared with an unpaced reference from the same
checkpoint.

One event is one accepted observation and its one free act. The clock supplies due
times to the driver and nothing to the brain. Sweeps, residual checks, refusals and
checkpoint IO are work. A refused act is a missed action that leaves state and trace
unchanged while the clock consumes the event.

## Run and verify

```sh
python benchmarks/rhythm/steady_rhythm.py --out /tmp/rhythm-run
python benchmarks/rhythm/steady_rhythm.py --verify /tmp/rhythm-run
python benchmarks/rhythm/report.py /tmp/rhythm-run/summary.json
```

The default run executes the frozen protocol on the five confirmation seeds, both
recipes and both arms (20 founders) in about 7 minutes on one laptop CPU, most of it
paced sleep. Any override (`--bouts`, `--window`, `--decay`, ...) marks the receipt
`frozen_protocol: false`. `--no-cadence` skips the paced and repair-speed forks. The
output directory holds the protocol copy, frozen inputs, every checkpoint,
`reports.jsonl` with every settlement and teaching report, `runs.jsonl` and the
`summary.json` receipt. `results/` keeps the receipts quoted below; checkpoints and
per-call reports stay out of the repository. `develop_recipe.py` reproduces the
development grid on the development seeds.

## Results on the five fresh seeds, 2026-10-04

Receipt: `results/confirmation-2026-10-04.json.gz` (a `cadence.Receipt`; its body is
read by `report.py`). Protocol SHA-256
`424a1604308725b3fa2e7708a947c6362150d5318d44f7f0cba39edef1e3a81f`, frozen. Every
planned founder completed; none was capped, refused an act or refused a lesson. The
tables below are printed by `report.py` from the stored per-event actions.

### Window controls, means over founders (alternation / agreement / refusals)

| Branch | compose_default/every | compose_default/mismatch | selected/every | selected/mismatch |
| --- | --- | --- | --- | --- |
| intact | 0.37 / 0.50 / 0 | 0.36 / 0.61 / 0 | 0.44 / 0.60 / 0 | 0.44 / 0.49 / 0 |
| restored | 0.37 / 0.50 / 0 | 0.36 / 0.61 / 0 | 0.44 / 0.60 / 0 | 0.44 / 0.49 / 0 |
| erased | 0.35 / 0.53 / 0 | 0.37 / 0.46 / 0 | 0.42 / 0.50 / 0 | 0.40 / 0.51 / 0 |
| shuffled | 0.37 / 0.50 / 0 | 0.36 / 0.41 / 0 | 0.44 / 0.51 / 0 | 0.44 / 0.50 / 0 |
| reset | 0.35 / 0.53 / 0 | 0.37 / 0.46 / 0 | 0.42 / 0.50 / 0 | 0.40 / 0.51 / 0 |
| static | 0.00 / 0.50 / 0 | 0.00 / 0.50 / 0 | 0.00 / 0.50 / 0 | 0.00 / 0.50 / 0 |
| static_cold | 0.00 / 0.50 / 0 | 0.00 / 0.50 / 0 | 0.00 / 0.50 / 0 | 0.00 / 0.50 / 0 |
| flipflop | 1.00 / 1.00 / 0 | 1.00 / 1.00 / 0 | 1.00 / 1.00 / 0 | 1.00 / 1.00 / 0 |
| random | 0.50 / 0.50 / 0 | 0.50 / 0.51 / 0 | 0.50 / 0.48 / 0 | 0.50 / 0.50 / 0 |
| shuffled_donor | 0.37 / 0.50 / 0 | 0.36 / 0.61 / 0 | 0.44 / 0.60 / 0 | 0.44 / 0.49 / 0 |

### Founders

| Recipe | Arm | Seed | intact alternation | intact agreement | blocks of 8 | erased | reset | static | flipflop | random |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| compose_default | every | 101 | 0.67 | 0.51 | 0.54 0.71 0.68 0.75 0.64 0.64 0.71 0.71 | 0.62 | 0.62 | 0.00 | 1.00 | 0.54 |
| compose_default | every | 102 | 0.00 | 0.50 | 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.50 |
| compose_default | every | 103 | 0.50 | 0.50 | 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 | 0.51 | 0.51 | 0.00 | 1.00 | 0.45 |
| compose_default | every | 104 | 0.67 | 0.49 | 0.68 0.68 0.64 0.68 0.68 0.64 0.68 0.68 | 0.60 | 0.60 | 0.00 | 1.00 | 0.48 |
| compose_default | every | 105 | 0.00 | 0.50 | 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.54 |
| compose_default | mismatch | 101 | 0.13 | 0.57 | 0.36 0.07 0.11 0.11 0.11 0.11 0.07 0.11 | 0.44 | 0.44 | 0.00 | 1.00 | 0.54 |
| compose_default | mismatch | 102 | 0.00 | 0.50 | 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.50 |
| compose_default | mismatch | 103 | 1.00 | 1.00 | 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 | 0.84 | 0.84 | 0.00 | 1.00 | 0.45 |
| compose_default | mismatch | 104 | 0.67 | 0.49 | 0.71 0.64 0.71 0.68 0.64 0.71 0.68 0.64 | 0.56 | 0.56 | 0.00 | 1.00 | 0.48 |
| compose_default | mismatch | 105 | 0.00 | 0.50 | 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.54 |
| selected | every | 101 | 0.54 | 0.37 | 0.57 0.61 0.57 0.57 0.64 0.57 0.61 0.57 | 0.56 | 0.56 | 0.00 | 1.00 | 0.54 |
| selected | every | 102 | 0.00 | 0.50 | 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.50 |
| selected | every | 103 | 1.00 | 1.00 | 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 | 0.95 | 0.95 | 0.00 | 1.00 | 0.45 |
| selected | every | 104 | 0.66 | 0.65 | 0.64 0.64 0.71 0.64 0.64 0.71 0.64 0.64 | 0.60 | 0.60 | 0.00 | 1.00 | 0.48 |
| selected | every | 105 | 0.00 | 0.50 | 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.54 |
| selected | mismatch | 101 | 0.84 | 0.49 | 0.86 0.86 0.86 0.86 0.86 0.86 0.86 0.86 | 0.79 | 0.79 | 0.00 | 1.00 | 0.54 |
| selected | mismatch | 102 | 0.00 | 0.50 | 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.50 |
| selected | mismatch | 103 | 0.73 | 0.49 | 0.68 0.79 0.68 0.75 0.75 0.79 0.68 0.71 | 0.63 | 0.63 | 0.00 | 1.00 | 0.45 |
| selected | mismatch | 104 | 0.65 | 0.49 | 0.46 0.71 0.64 0.71 0.75 0.61 0.68 0.75 | 0.56 | 0.56 | 0.00 | 1.00 | 0.48 |
| selected | mismatch | 105 | 0.00 | 0.50 | 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.54 |

### Disturbances, means over founders

| Disturbance | Model | pre alternation | post alternation | agreement hold | agreement continue | recovered rows / rows |
| --- | --- | --- | --- | --- | --- | --- |
| distractor | compose_default/every brain | 0.34 | 0.37 | 0.47 | 0.57 | 0/20 |
| distractor | flipflop | 1.00 | 1.00 | 0.00 | 1.00 | 20/20 |
| distractor | random | 0.44 | 0.52 | 0.50 | 0.49 | 0/20 |
| pause1 | compose_default/every brain | 0.34 | 0.36 | 0.47 | 0.57 | 0/20 |
| pause1 | flipflop | 1.00 | 1.00 | 0.00 | 1.00 | 20/20 |
| pause1 | random | 0.49 | 0.48 | 0.51 | 0.52 | 0/20 |
| pause2 | compose_default/every brain | 0.34 | 0.36 | 0.53 | 0.53 | 0/20 |
| pause2 | flipflop | 1.00 | 1.00 | 1.00 | 1.00 | 20/20 |
| pause2 | random | 0.55 | 0.52 | 0.49 | 0.49 | 0/20 |
| pause4 | compose_default/every brain | 0.34 | 0.35 | 0.52 | 0.51 | 0/20 |
| pause4 | flipflop | 1.00 | 1.00 | 1.00 | 1.00 | 20/20 |
| pause4 | random | 0.50 | 0.52 | 0.44 | 0.49 | 0/20 |
| distractor | compose_default/mismatch brain | 0.41 | 0.43 | 0.37 | 0.62 | 4/20 |
| distractor | flipflop | 1.00 | 1.00 | 0.00 | 1.00 | 20/20 |
| distractor | random | 0.44 | 0.52 | 0.50 | 0.49 | 0/20 |
| pause1 | compose_default/mismatch brain | 0.41 | 0.42 | 0.37 | 0.61 | 3/20 |
| pause1 | flipflop | 1.00 | 1.00 | 0.00 | 1.00 | 20/20 |
| pause1 | random | 0.49 | 0.48 | 0.51 | 0.52 | 0/20 |
| pause2 | compose_default/mismatch brain | 0.41 | 0.36 | 0.59 | 0.61 | 0/20 |
| pause2 | flipflop | 1.00 | 1.00 | 1.00 | 1.00 | 20/20 |
| pause2 | random | 0.55 | 0.52 | 0.49 | 0.49 | 0/20 |
| pause4 | compose_default/mismatch brain | 0.41 | 0.36 | 0.60 | 0.60 | 4/20 |
| pause4 | flipflop | 1.00 | 1.00 | 1.00 | 1.00 | 20/20 |
| pause4 | random | 0.50 | 0.52 | 0.44 | 0.49 | 0/20 |
| distractor | selected/every brain | 0.44 | 0.44 | 0.49 | 0.65 | 4/20 |
| distractor | flipflop | 1.00 | 1.00 | 0.00 | 1.00 | 20/20 |
| distractor | random | 0.44 | 0.52 | 0.50 | 0.49 | 0/20 |
| pause1 | selected/every brain | 0.44 | 0.45 | 0.49 | 0.62 | 4/20 |
| pause1 | flipflop | 1.00 | 1.00 | 0.00 | 1.00 | 20/20 |
| pause1 | random | 0.49 | 0.48 | 0.51 | 0.52 | 0/20 |
| pause2 | selected/every brain | 0.44 | 0.44 | 0.51 | 0.69 | 4/20 |
| pause2 | flipflop | 1.00 | 1.00 | 1.00 | 1.00 | 20/20 |
| pause2 | random | 0.55 | 0.52 | 0.49 | 0.49 | 0/20 |
| pause4 | selected/every brain | 0.44 | 0.45 | 0.52 | 0.57 | 4/20 |
| pause4 | flipflop | 1.00 | 1.00 | 1.00 | 1.00 | 20/20 |
| pause4 | random | 0.50 | 0.52 | 0.44 | 0.49 | 0/20 |
| distractor | selected/mismatch brain | 0.40 | 0.43 | 0.50 | 0.50 | 0/20 |
| distractor | flipflop | 1.00 | 1.00 | 0.00 | 1.00 | 20/20 |
| distractor | random | 0.44 | 0.52 | 0.50 | 0.49 | 0/20 |
| pause1 | selected/mismatch brain | 0.40 | 0.43 | 0.50 | 0.50 | 0/20 |
| pause1 | flipflop | 1.00 | 1.00 | 0.00 | 1.00 | 20/20 |
| pause1 | random | 0.49 | 0.48 | 0.51 | 0.52 | 0/20 |
| pause2 | selected/mismatch brain | 0.40 | 0.44 | 0.49 | 0.50 | 0/20 |
| pause2 | flipflop | 1.00 | 1.00 | 1.00 | 1.00 | 20/20 |
| pause2 | random | 0.55 | 0.52 | 0.49 | 0.49 | 0/20 |
| pause4 | selected/mismatch brain | 0.40 | 0.44 | 0.50 | 0.48 | 0/20 |
| pause4 | flipflop | 1.00 | 1.00 | 1.00 | 1.00 | 20/20 |
| pause4 | random | 0.50 | 0.52 | 0.44 | 0.49 | 0/20 |

### Cadence, repair speed and host load

| Variant | founders identical to reference | refusals | sweeps per event | solve ms mean / max | lateness ms max | missed deadlines | slot agreement |
| --- | --- | --- | --- | --- | --- | --- | --- |
| budget_16 | 20/20 | 0 | 5.5 | unpaced | unpaced | unpaced | unpaced |
| budget_256 | 20/20 | 0 | 20.4 | unpaced | unpaced | unpaced | unpaced |
| budget_64 | 20/20 | 0 | 20.4 | unpaced | unpaced | unpaced | unpaced |
| paced_extra | 20/20 | 0 | 20.4 | 1.07 / 2.26 | 10.91 | 0 | 0.47 |
| paced_regular | 20/20 | 0 | 20.4 | 0.89 / 12.31 | 10.05 | 0 | 0.55 |
| paced_regular200 | 20/20 | 0 | 20.4 | 1.07 / 2.72 | 89.97 | 0 | 0.55 |
| paced_regular50 | 20/20 | 0 | 20.4 | 0.98 / 1.98 | 10.09 | 0 | 0.55 |
| paced_regular_load | 20/20 | 0 | 20.4 | 0.87 / 140.57 | 47.51 | 1 | 0.55 |
| paced_skipped | 20/20 | 0 | 20.5 | 1.09 / 2.44 | 10.11 | 0 | 0.48 |
| reference | 20/20 | 0 | 20.4 | unpaced | unpaced | unpaced | unpaced |
| tolerance_0.0001 | 20/20 | 0 | 22.4 | unpaced | unpaced | unpaced | unpaced |
| tolerance_0.01 | 20/20 | 0 | 19.8 | unpaced | unpaced | unpaced | unpaced |

### Custody and work

- shuffled rows reproducing the donor row's intact sequence: 80/80; erased equals reset in 20/20 founders; window cue followed in 41/80 rows
- founders planned 20, completed 20, capped False, run seconds 425
- probe continuation equal (actions and saved arrays): 20/20
- mid-pause continuation equal: 20/20
- protocol sha256 424a1604308725b3fa2e7708a947c6362150d5318d44f7f0cba39edef1e3a81f, frozen True
- compose_default/every: lessons 1440, refused 0, free budget exhausted 0, last-bout agreement with the label 0.38, flip-flop lessons 5760
- compose_default/mismatch: lessons 1205, refused 0, free budget exhausted 0, last-bout agreement with the label 0.43, flip-flop lessons 130
- selected/every: lessons 1440, refused 0, free budget exhausted 0, last-bout agreement with the label 0.45, flip-flop lessons 5760
- selected/mismatch: lessons 1246, refused 0, free budget exhausted 0, last-bout agreement with the label 0.46, flip-flop lessons 130

## Reading the controls

`static_cold` is the instrument's zero: a brain reset before every act under an
identical drive gives the same action at every event. `static` adds the warm neural
state and shows whether that state alone carries phase; it scores zero as well, so
the trace is necessary for any alternation. `erased` and `reset` keep the learned
parameters and lose the phase once; their action sequences are identical in every
founder, so the warm neural state contributes nothing once the trace is gone, and
the alternation that resumes from them is re-established by the parameters with the
pre-probe phase lost (agreement near 0.5). Every `shuffled` row reproduces the donor
row's intact sequence event for event: the transplanted trace carries the donor's
phase and determines the whole trajectory. `flipflop` is what an explicit recurrence
over the previous output does with the same lessons: alternation 1.00 in every
founder, under every disturbance. `random` has an expected alternation rate of 0.5.

Transient cue-following and continuing alternation are separated by the blocks of 8
events in the 64-event window. The founders that alternate hold their rate through
the eighth block; the founders that hold one action do so from the first block. The
teaching-phase agreement with the label (0.38 to 0.46 in the last bout) is reported
separately from the free window.

## Limits

The measured System 1 holds one action in two of five founders at both recipes and
both arms, alternates with period 2 in one founder (seed 103) under three of the four
recipe/arm combinations, and alternates in 54 to 84 percent of event pairs in the
others. The mean across founders, with every founder in the denominator, is 0.44 for
the selected recipe and 0.37 for the compose default, both below the uniform-random
baseline of 0.50 and far below the flip-flop control. Disturbances change little
because the rhythm that exists is fragile before the disturbance; the flip-flop
recovers in 20 of 20 rows after every disturbance, the brain in 0 to 4. Four
rows share one set of parameters; a founder with another seed is another brain. The
paced measurements are from one laptop and describe its sleep granularity and solve
time, not an embedded clock. Identical per-event actions across cadences, budgets,
tolerances and host load follow from a deterministic CPU solve with no hidden thread;
they establish event-time semantics, not robustness to a solver that fails to
qualify, and no budget down to 16 sweeps produced a refusal on this 68-neuron graph.
Issue 93 consumes a passed continuing-action seam; this chamber reports a failed
one for the current System 1 and the instrument to measure a repair.

## rhythm/2: the efference copy, 2026-10-08

The rhythm/1 result above is the measured limit of a System 1 whose only carried
history is the working trace, a copy of the settled association state taken before
the decision. Under identical drive that state records which action was executed
only through the margin the motor competition left, and the flip-flop control that
scores 1.00 is handed its own previous action explicitly. rhythm/2 declares one
mechanism against that limit: the efference copy, `Brain.compose(...,
efference_amplitude=3.0, efference_decay=0.0)`, one `efference` neuron per motor
neuron driven by the one-hot of the command the row issued at the previous event
and read by the association region through a plastic projection at the working
trace's scale. The write is fixed, like the working trace's; what to do after what
it did is learned. Nothing else changes: the `efference` recipe is the rhythm/1
`selected` recipe plus the copy, and `selected` runs on every founder as the
control, byte-identical to rhythm/1. The copy is a gene with zero as its control,
not a default. [`protocol-2.json`](protocol-2.json), SHA-256
`017b7f8b2c9ea53cc97a04f2fbf854f844158bcb73610e1f1723c14d088b87a4`, keeps every
other declaration of rhythm/1, declares the carried phase state as the working
trace and the copy, and lets the `erased`, `shuffled`, `static` and `reset` controls
act on both. Its confirmation seeds 201 to 205 are fresh.

```sh
python benchmarks/rhythm/steady_rhythm.py --out /tmp/rhythm-2 \
    --protocol benchmarks/rhythm/protocol-2.json --recipes efference selected
python benchmarks/rhythm/steady_rhythm.py --verify /tmp/rhythm-2
python benchmarks/rhythm/report.py /tmp/rhythm-2/summary.json
python benchmarks/rhythm/develop_recipe.py --out /tmp/rhythm-dev --decays 0.1 \
    --amplitudes 3.0 --rates default --efference-amplitudes 0.0 0.3 1.0 3.0 \
    --efference-decays 0.0 0.2 0.5
```

The development command includes the amplitude-zero control once for each decay;
those control cells are identical because decay has no effect without a copy.
The archived table keeps one copy of each amplitude-zero control cell.

### Development, seeds 0 to 5, 24 bouts

`results/development-efference.json` keeps every cell, the amplitude-0 control
included. Mean window alternation over the six development founders:

| efference amplitude | decay | every | mismatch |
| --- | --- | --- | --- |
| 0 (the rhythm/1 selected brain) | | 0.78 | 0.69 |
| 0.3 | 0.0 / 0.2 / 0.5 | 0.63 / 0.75 / 0.82 | 0.55 / 0.65 / 0.61 |
| 1.0 | 0.0 / 0.2 / 0.5 | 0.84 / 0.85 / 0.60 | 0.66 / 0.72 / 0.56 |
| 3.0 | 0.0 / 0.2 / 0.5 | **1.00** / 0.85 / 0.74 | **0.93** / 0.86 / 0.74 |

The control reproduces the rhythm/1 development figure (0.73 over both arms).
Amplitude 3.0 at decay 0.0, the one-hot of the last command and nothing older,
alternates at 1.00 on all six founders in the `every` arm and was frozen as the
candidate; a slower decay or a weaker read is worse.

### Confirmation on the five fresh seeds, 2026-10-08

Receipt: `results/confirmation-2-2026-10-08.json.gz`, verified (canonical form,
digest, sources and artifact hashes agree), frozen protocol, 20 founders planned and
completed, none capped, no refused act or lesson, 442 seconds. The tables are
printed by `report.py` from the stored per-event actions. This archived receipt is
bound to the sources at commit `1b916fb`, before the maintainer fixes; its source
hashes describe that revision and remain unchanged.

Maintainer verification: `results/confirmation-2-maintainer-2026-10-08.json.gz`
repeats the frozen protocol on the runtime sources at `3c64122`, after the fixes.
All 20 founders completed in 440 seconds; canonical form, digest, sources and
artifact hashes verify. All 960 recorded action sequences (165,520 row-events),
scores and solver-work counters match the original, as do all 5,515 arrays across
185 saved artifacts. Timing was remeasured, with zero missed deadlines against
one in the original run. The tables below retain the original measurements.

#### Window controls, means over founders (alternation / agreement / refusals)

| Branch | efference/every | efference/mismatch | selected/every | selected/mismatch |
| --- | --- | --- | --- | --- |
| intact | **0.97** / 0.89 / 0 | **0.93** / 0.78 / 0 | 0.50 / 0.55 / 0 | 0.40 / 0.50 / 0 |
| restored | 0.97 / 0.89 / 0 | 0.93 / 0.78 / 0 | 0.50 / 0.55 / 0 | 0.40 / 0.50 / 0 |
| erased | 0.97 / 0.59 / 0 | 0.90 / 0.49 / 0 | 0.47 / 0.52 / 0 | 0.41 / 0.48 / 0 |
| shuffled | 0.97 / 0.50 / 0 | 0.93 / 0.46 / 0 | 0.50 / 0.52 / 0 | 0.40 / 0.49 / 0 |
| reset | 0.97 / 0.59 / 0 | 0.90 / 0.49 / 0 | 0.48 / 0.52 / 0 | 0.42 / 0.47 / 0 |
| static | 0.00 / 0.50 / 0 | 0.00 / 0.50 / 0 | 0.00 / 0.50 / 0 | 0.00 / 0.50 / 0 |
| static_cold | 0.00 / 0.50 / 0 | 0.00 / 0.50 / 0 | 0.00 / 0.50 / 0 | 0.00 / 0.50 / 0 |
| flipflop | 1.00 / 1.00 / 0 | 1.00 / 1.00 / 0 | 1.00 / 1.00 / 0 | 1.00 / 1.00 / 0 |
| random | 0.50 / 0.53 / 0 | 0.50 / 0.48 / 0 | 0.50 / 0.48 / 0 | 0.50 / 0.49 / 0 |
| shuffled_donor | 0.97 / 0.89 / 0 | 0.93 / 0.78 / 0 | 0.50 / 0.55 / 0 | 0.40 / 0.50 / 0 |

#### Founders

| Recipe | Arm | Seed | intact alternation | intact agreement | blocks of 8 | erased | reset | static | flipflop | random |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| efference | every | 201 | 1.00 | 0.98 | 1.00 1.00 1.00 1.00 1.00 1.00 1.00 0.96 | 1.00 | 1.00 | 0.00 | 1.00 | 0.50 |
| efference | every | 202 | 1.00 | 1.00 | 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 | 1.00 | 1.00 | 0.00 | 1.00 | 0.51 |
| efference | every | 203 | 1.00 | 1.00 | 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 | 1.00 | 1.00 | 0.00 | 1.00 | 0.53 |
| efference | every | 204 | 1.00 | 1.00 | 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 | 0.97 | 0.97 | 0.00 | 1.00 | 0.43 |
| efference | every | 205 | 0.86 | 0.48 | 0.82 0.86 0.89 0.79 0.93 0.82 0.93 0.86 | 0.89 | 0.89 | 0.00 | 1.00 | 0.54 |
| efference | mismatch | 201 | 0.74 | 0.59 | 0.79 0.75 0.71 0.75 0.79 0.75 0.79 0.75 | 0.79 | 0.79 | 0.00 | 1.00 | 0.50 |
| efference | mismatch | 202 | 0.90 | 0.52 | 0.93 0.93 0.86 0.89 0.96 0.93 0.89 0.86 | 0.89 | 0.89 | 0.00 | 1.00 | 0.51 |
| efference | mismatch | 203 | 1.00 | 1.00 | 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 | 0.98 | 0.98 | 0.00 | 1.00 | 0.53 |
| efference | mismatch | 204 | 1.00 | 1.00 | 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 | 0.87 | 0.87 | 0.00 | 1.00 | 0.43 |
| efference | mismatch | 205 | 1.00 | 0.79 | 1.00 0.96 1.00 1.00 1.00 1.00 1.00 1.00 | 0.95 | 0.95 | 0.00 | 1.00 | 0.54 |
| selected | every | 201 | 0.64 | 0.52 | 0.54 0.68 0.54 0.68 0.54 0.71 0.75 0.64 | 0.63 | 0.70 | 0.00 | 1.00 | 0.50 |
| selected | every | 202 | 0.00 | 0.50 | 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 | 0.08 | 0.08 | 0.00 | 1.00 | 0.51 |
| selected | every | 203 | 0.71 | 0.68 | 0.71 0.68 0.68 0.71 0.68 0.68 0.71 0.68 | 0.48 | 0.48 | 0.00 | 1.00 | 0.53 |
| selected | every | 204 | 0.62 | 0.51 | 0.57 0.61 0.57 0.64 0.64 0.64 0.57 0.64 | 0.60 | 0.60 | 0.00 | 1.00 | 0.43 |
| selected | every | 205 | 0.56 | 0.52 | 0.46 0.36 0.50 0.43 0.68 0.71 0.54 0.79 | 0.56 | 0.56 | 0.00 | 1.00 | 0.54 |
| selected | mismatch | 201 | 0.46 | 0.50 | 0.61 0.39 0.36 0.32 0.57 0.57 0.50 0.46 | 0.53 | 0.54 | 0.00 | 1.00 | 0.50 |
| selected | mismatch | 202 | 0.00 | 0.50 | 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 | 0.24 | 0.24 | 0.00 | 1.00 | 0.51 |
| selected | mismatch | 203 | 0.49 | 0.50 | 0.43 0.43 0.43 0.43 0.43 0.43 0.43 0.43 | 0.25 | 0.25 | 0.00 | 1.00 | 0.53 |
| selected | mismatch | 204 | 0.46 | 0.54 | 0.39 0.50 0.46 0.54 0.46 0.43 0.46 0.39 | 0.44 | 0.44 | 0.00 | 1.00 | 0.43 |
| selected | mismatch | 205 | 0.58 | 0.48 | 0.75 0.68 0.46 0.54 0.43 0.50 0.71 0.54 | 0.60 | 0.60 | 0.00 | 1.00 | 0.54 |

#### Disturbances, means over founders

| Disturbance | Model | pre alternation | post alternation | agreement hold | agreement continue | recovered rows / rows |
| --- | --- | --- | --- | --- | --- | --- |
| distractor | efference/every brain | 0.96 | 0.97 | 0.10 | 0.90 | 16/20 |
| pause1 | efference/every brain | 0.96 | 0.97 | 0.08 | 0.93 | 16/20 |
| pause2 | efference/every brain | 0.96 | 0.98 | 0.90 | 0.93 | 16/20 |
| pause4 | efference/every brain | 0.96 | 0.98 | 0.91 | 0.92 | 16/20 |
| distractor | efference/mismatch brain | 0.94 | 0.92 | 0.27 | 0.75 | 11/20 |
| pause1 | efference/mismatch brain | 0.94 | 0.93 | 0.24 | 0.77 | 11/20 |
| pause2 | efference/mismatch brain | 0.94 | 0.94 | 0.77 | 0.78 | 11/20 |
| pause4 | efference/mismatch brain | 0.94 | 0.93 | 0.72 | 0.77 | 12/20 |
| distractor | selected/every brain | 0.46 | 0.49 | 0.44 | 0.54 | 1/20 |
| pause1 | selected/every brain | 0.46 | 0.50 | 0.44 | 0.53 | 1/20 |
| pause2 | selected/every brain | 0.46 | 0.47 | 0.53 | 0.51 | 1/20 |
| pause4 | selected/every brain | 0.46 | 0.50 | 0.51 | 0.50 | 1/20 |
| distractor | selected/mismatch brain | 0.44 | 0.39 | 0.53 | 0.50 | 0/20 |
| pause1 | selected/mismatch brain | 0.44 | 0.42 | 0.50 | 0.48 | 0/20 |
| pause2 | selected/mismatch brain | 0.44 | 0.41 | 0.48 | 0.50 | 0/20 |
| pause4 | selected/mismatch brain | 0.44 | 0.39 | 0.47 | 0.51 | 0/20 |
| every disturbance | flipflop | 1.00 | 1.00 | 0.00 (odd) / 1.00 (even) | 1.00 | 20/20 |
| every disturbance | random | 0.46 to 0.50 | 0.49 to 0.52 | 0.50 to 0.54 | 0.47 to 0.53 | 0/20 |

#### Cadence, repair speed and host load

Per-event actions are identical to the unpaced reference in 20 of 20 founders at
100, 50 and 200 ms per event, with an extra event in slot 10, with slot 10 skipped,
beside one spinning process per logical CPU (one missed deadline, in a `selected`
founder, with identical actions), at free-step budgets 256 and 64 and at tolerances
0.01 and 0.0001; zero refusals in every one of those variants. At a budget of 16
sweeps, two founders differ from the reference and 128 row-events refuse, all of them
in `efference/every`: the copy's drive makes a free solve that needs more than 16
sweeps on those founders. Every act of both recipes takes 32 sweeps per event at the
protocol's budget of 1024 (the solver checks the residual after 32 sweeps and
qualifies); the rhythm/1 receipt recorded 20.4 on its library of 2026-10-04, and the
`selected` recipe takes the same 32 sweeps per act on the released 0.75.0 wheel, on
main and on this branch, so the difference is the library's since then, not the copy's.

#### Custody and work

- shuffled rows reproducing the donor row's intact sequence: 80/80; erased equals
  reset in 18/20 founders; window cue followed in 41/80 rows
- probe continuation equal (actions and saved arrays): 20/20; mid-pause
  continuation equal: 20/20
- efference/every: lessons 1440, refused 0, free budget exhausted 0, last-bout
  agreement with the label 0.92 (selected/every: 0.57); flip-flop lessons 5760
- efference/mismatch: lessons 559, refused 0, last-bout agreement 0.89
  (selected/mismatch: 1214 lessons, 0.46); flip-flop lessons 134
- teaching sweeps per founder, `every` arm: 2,035 to 3,531 with the copy against
  2,035 to 2,495 without; a lesson on the copy's drive costs more sweeps

### Reading the rhythm/2 controls

`static` and `static_cold` score zero for the copy as for the trace: with no carried
history the brain holds one action, so the carried history is the whole carrier of
the beat. `erased` and `reset` lose the phase once and resume alternating from the
next event (agreement with the continued ideal 0.49 to 0.59, alternation 0.90 to
0.97): at decay 0 the copy is rewritten in full by the next act, so erasing it costs
one event of phase and nothing of the rhythm, which lives in the learned projection
from the copy. Every `shuffled` row reproduces the donor row's intact sequence event
for event, so the transplanted history determines the trajectory. After a distractor
or a one-event pause the brain continues the beat through the event (agreement with
`continue` 0.90 to 0.93, with `hold` 0.08 to 0.10), as the flip-flop does; after a
pause of two or four events both hypotheses coincide. Recovery to uninterrupted
alternation: 16 of 20 rows in the `every` arm and 11 to 12 in `mismatch`, against 0
to 1 for the control and 20 of 20 for the flip-flop.

### Failures and limits of rhythm/2

- Seed 205 in the `every` arm alternates in 86 percent of event pairs and holds the
  cue in 48 percent of rows; seed 201 in `mismatch` alternates in 74 percent. The
  mean with every founder in the denominator is 0.97 and 0.93, not 1.00; the
  flip-flop is 1.00 on every founder.
- The control holds one action in seed 202 under both arms and alternates in 46 to
  71 percent of pairs elsewhere, means 0.50 and 0.40: the rhythm/1 limit again on
  five new founders.
- The copy costs work: more teaching sweeps per founder and refusals at a 16-sweep
  free budget where the control had none. It is one more bounded state per stream
  and one more plastic projection; it is not free.
- The copy at decay 0 carries one event of history. A longer sequence, a pause the
  brain should wait through rather than step through, and the reward-driven version
  of the task are not measured here; neither are more actions, more modules or
  observers. Integration with issue 93 stays open.
- Four rows share one set of parameters; a founder with another seed is another
  brain. The paced measurements are from one laptop.

Against the acceptance boxes of issue 116: box 2's learning half, which rhythm/1
reported failed, now passes on the declared window for four of five fresh founders
in the `every` arm and three of five in `mismatch`, with the fifth and the other two
above 0.74; boxes 3 and 4 are measured with the same instrument, with recovery
present for the brain and complete for the control; box 5's custody passes on five
fresh seeds and the timing envelope is confirmed at budgets of 64 sweeps and above.
The copy is the declared mechanism of this result, selected on the development
seeds and confirmed once on fresh seeds; no default changes.

## The beat paid by the world: reward/1, reward/2 and reward/3, 2026-10-08

rhythm/1 and rhythm/2 teach the beat with a label at every event. The manifesto's
creature learns from one reward channel over its own actions, and the world supplies
problems, never answers. [`reward_rhythm.py`](reward_rhythm.py) is that version of the
task: one continuing `Brain.live` life, the same drive at every moment, no label, no
clock; a step on the other foot than the last one earns one unit, paid at the next
moment as the outcome of that step, and a repeated step earns nothing. Nothing in the
observation says which foot moved last; no associative memory can key on a cue that
never changes; the efference copy carries the walker's own last command, alongside its working trace.
The copyless arm tests whether that command boundary helps the graph's reward learning. The walker is aroused
and learning through a youth of 100 moments, then routine while outcomes match its
forecasts and aroused again when they do not or when its need goes unmet.

Arms on the same founder weights: `live` (the declared System 1 with the copy),
`nocopy` (the same brain without it), `defaults` (the copy on the composed reward
defaults instead of the declared operating point), `step` (always learning, every
moment sampled), `lambda_control` (the eligibility decay at its control value),
`frozen` (no learning: the founder's greedy choice, which shows whether a founder had
the beat from birth), `tabular` (Q-learning given the one bit the copy carries) and
`random`. At moment 400 the live brain is saved with its last action awaiting its
outcome; a loaded twin lives beside it for 64 moments under the same outcomes (equal
actions and equal saved arrays are the custody check), a loaded copy acts greedily for
32 moments (the acquired policy without exploration), and forks of the checkpoint meet
a pause of one, two or four silent events or one distractor event and keep living under
the same rule. The scored window is the last 64 moments of a 600-moment life, still
paid. A refused answer is a missed step that earns nothing. Every moment of every arm,
every learning phase, probe, twin, fork and checkpoint is charged.

```sh
python benchmarks/rhythm/reward_rhythm.py --out /tmp/reward-2 \
    --protocol benchmarks/rhythm/protocol-reward-2.json
python benchmarks/rhythm/reward_rhythm.py --verify /tmp/reward-2
```

The original receipts and tables below are preserved as historical measurements
from chamber source `639bc83` and Cadence 0.76.0. The maintainer review found that
disturbance forks discarded the pending reward and previous action at their saved
boundary, refusal retries could lose or repeat feedback, and work accounting omitted
some checkpoint and probe costs. Historical disturbance and work figures therefore
describe that instrument. Successful main-life action sequences remain useful bounded
evidence; the recorded confirmations contain no refusals. Corrected runs carry an
explicit instrument revision and separate receipts. Gate checks also require the
declared control and the same founders to satisfy all criteria.

### Development, seeds 0 to 5

The operating point is the key-door nursery's declared point (`benchmarks/keydoor`:
working trace 0.3, actor rate 0.1, eligibility decay 0.95, discount 0.95, critic rate
5) with the rhythm/2 efference genes (amplitude 3.0, decay 0.0) and the nursery's
arousal genes with a youth of 100 moments and a need of 0.5, half a perfect walker's
income. Every variant's receipt is in `results/development-reward-*.json.gz`; mean
window alternation over six founders, with the greedy probe and the share of the window
aroused for the brain arms:

| variant | live | nocopy | defaults | step | eligibility 0 | frozen | tabular | random |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| need 0.5 (declared) | **1.00** / 1.00 / 0.00 | 0.46 / 0.00 / 0.82 | 0.82 / 0.67 / 0.31 | 0.98 / 1.00 / 1.00 | 1.00 / 1.00 / 0.00 | 0.00 | 0.95 | 0.50 |
| need 0 | 1.00 / 1.00 / 0.00 | 0.02 / 0.00 / 0.03 | 0.78 / 0.77 / 0.00 | 0.98 / 1.00 / 1.00 | 0.78 / 0.78 / 0.00 | 0.00 | 0.95 | 0.50 |
| youth 0 (born calm) | 0.83 / 0.83 / 0.00 | 0.46 / 0.00 / 0.86 | 0.82 / 0.67 / 0.27 | 0.98 / 1.00 / 1.00 | 0.94 / 0.94 / 0.00 | 0.00 | 0.95 | 0.50 |
| working trace 3.0 (the supervised recipe) | 0.62 / 0.44 / 0.17 | 0.61 / 0.45 / 0.59 | 0.59 / 0.22 / 0.74 | 0.34 / 0.53 / 1.00 | 0.63 / 0.47 / 0.29 | 0.18 | 0.95 | 0.50 |

At the declared point all six founders learn the beat from reward alone within their
youth, walk it for the remaining 500 moments without a single aroused moment, and their
greedy probes walk it too; the frozen founders hold one foot, so the beat was learned.
Without the copy the walker alternates at 0.46 and stays restless (need 0.5), or
calms into mostly holding one foot (0.02 alternation at need 0): an internally coherent equilibrium that is wrong about the
world, and calm about it. Born calm, three founders settle into a limp of two changed
steps in three, which pays more than the need and so never rouses them. The supervised
chamber's strong working trace degrades every learning arm: where the previous
moment's settled state carries nothing the decision needs, the trace is interference the
actor must first learn away.

The second-freeze candidates (`development-reward-need09`, `-lam0`, `-need09lam0`:
a need of 0.9, the eligibility decay at 0, and both) each give live 1.00 on all six
founders, calm, with the frozen founders at 0.00, exactly as the declared point did:
development does not discriminate between them. The fresh seeds did.

### reward/1: confirmation on fresh seeds 301 to 305, gate failed

[`protocol-reward.json`](protocol-reward.json), SHA-256
`f78c3f81f0c5d22c980d3e650042d04c376ea6b11bea0e24fc9bd00a184cddef`; receipt
`results/confirmation-reward-2026-10-08.json.gz`, verified, 40 founders planned and
completed, none capped, no refused answer, 32 seconds. Cells are window alternation /
greedy-probe alternation / share of the window aroused.

| seed | live | eligibility 0 | defaults | frozen | nocopy | step | tabular | random |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 301 | **1.00** / 1.00 / 0.00 | 1.00 / 1.00 / 0.00 | 0.56 / 0.00 / 0.81 | 0.00 / 0.00 / 0.00 | 0.41 / 0.00 / 0.84 | 1.00 / 1.00 / 1.00 | 0.97 | 0.48 |
| 302 | 1.00 / 1.00 / 0.00 | 1.00 / 1.00 / 0.00 | 1.00 / 1.00 / 0.00 | **1.00** / 1.00 / 0.00 | 0.40 / 0.00 / 0.73 | 0.97 / 1.00 / 1.00 | 0.95 | 0.30 |
| 303 | **1.00** / 1.00 / 0.00 | 1.00 / 1.00 / 0.00 | 1.00 / 1.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.41 / 0.00 / 0.84 | 0.98 / 1.00 / 1.00 | 0.95 | 0.44 |
| 304 | 0.67 / 0.65 / 0.00 | 1.00 / 1.00 / 0.00 | 1.00 / 1.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.48 / 0.00 / 0.86 | 1.00 / 1.00 / 1.00 | 0.95 | 0.49 |
| 305 | 0.67 / 0.65 / 0.00 | 1.00 / 1.00 / 0.00 | 1.00 / 1.00 / 0.00 | **1.00** / 1.00 / 0.00 | 0.40 / 0.00 / 0.77 | 0.97 / 1.00 / 1.00 | 0.98 | 0.40 |

The declared gate asked four of five founders to alternate in at least 0.9 of the
window: three did (3/5 acquired, 3/5 greedy, 5/5 calm, 5/5 continued), so the gate
failed. Of the three, seed 302 had the beat from birth: its frozen founder, the greedy
choice without any learning, alternates at 1.00, as seed 305's does, because the copy's
random projection into the association region can make a two-cycle on its own. Seeds
304 and 305 ended in a limp, `RLLRLLRLL`, two changed steps in three, paid 0.67 per
moment, above the need of 0.5, and calm in it. Two of five fresh founders learned the
beat; reward/1 stands as a failed freeze. The eligibility-0 arm reached 1.00 on all five
founders on these seeds, which is the hypothesis reward/2 declares; the composed
defaults reached 4/5 and the frozen founders 2/5 from birth, which is why reward/2
gates learning, not acquisition.

### reward/2: confirmation on fresh seeds 401 to 405, gate passed

[`protocol-reward-2.json`](protocol-reward-2.json), SHA-256
`e942a1779f2da52e8b72858cc7839d4e5adabca0796183ea7a0f6b2866975366`: reward/1's point
with the eligibility decay at 0 (the world pays a step at the next moment, so the credit
belongs to that step alone), reward/1's decay of 0.95 as the `lambda_control` arm, the
same need, youth and world, fresh seeds, and a gate on founders that *learned* the beat:
live at or above 0.9 while the same founder's frozen arm is below it. Receipt
`results/confirmation-reward-2-2026-10-08.json.gz`, verified, 40 founders planned and
completed, none capped, no refused answer, 21 seconds.

| seed | live | eligibility 0.95 (control) | defaults | frozen | nocopy | step | tabular | random |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 401 | **1.00** / 1.00 / 0.00 | 1.00 / 1.00 / 0.00 | 1.00 / 1.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.41 / 0.00 / 0.73 | 1.00 / 1.00 / 1.00 | 0.97 | 0.57 |
| 402 | **1.00** / 1.00 / 0.00 | 1.00 / 1.00 / 0.00 | 1.00 / 1.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.49 / 0.00 / 0.72 | 0.84 / 1.00 / 1.00 | 0.95 | 0.54 |
| 403 | 0.67 / 0.68 / 0.00 | 1.00 / 1.00 / 0.00 | 0.67 / 0.68 / 0.00 | 0.00 / 0.00 / 0.00 | 0.37 / 0.00 / 0.72 | 1.00 / 1.00 / 1.00 | 0.90 | 0.43 |
| 404 | **1.00** / 1.00 / 0.00 | 0.67 / 0.68 / 0.00 | 0.51 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.41 / 0.00 / 0.89 | 0.97 / 1.00 / 1.00 | 0.95 | 0.52 |
| 405 | **1.00** / 1.00 / 0.00 | 1.00 / 1.00 / 0.00 | 1.00 / 1.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.46 / 0.00 / 0.88 | 1.00 / 1.00 / 1.00 | 0.97 | 0.51 |

Gates: learned 4/5, greedy 4/5, calm 5/5, continued 5/5, against 4 of 5 required:
**passed**. No fresh founder had the beat from birth (frozen 0.00 on all five). The
four that learned it did so within their youth or soon after (aroused in 0.40 of the
second block of 100 moments, 0.03 of the third, none afterwards), walk it for the rest
of the life at routine cost, and their greedy probes walk it. Seed 403 limps, calm, at
0.67. The control at the old eligibility decay also learns in 4/5 and limps on a
different founder (404), so the decay is not shown to be the cause of the limp; the
composed reward defaults learn in 3/5; the walker without the copy earns occasional
rewards and stays restless on every founder (0.37 to 0.49 alternation, aroused 0.72 to
0.89 of the window); the
always-learning loop reaches 0.96 while aroused at every moment; the tabular learner
given the same one bit reaches 0.90 to 0.97; uniform random 0.43 to 0.57.

Means over the five founders (window alternation / greedy / aroused share / income):
live 0.93 / 0.94 / 0.00 / 0.93; control 0.93 / 0.94 / 0.00 / 0.93; defaults 0.83 / 0.74 /
0.00 / 0.83; frozen 0.00; nocopy 0.43 / 0.00 / 0.79 / 0.43; step 0.96 / 1.00 / 1.00 /
0.96; tabular 0.95; random 0.51. Recovery to uninterrupted alternation after a pause of
one, two or four silent events or a distractor: live 3/5 founders, control 4/5, defaults
3/5, step 2/5, frozen and nocopy 0/5. Twins continue identically in 5/5 founders of every
brain arm. Work per moment over a life: about 33 sweeps for the walker with the copy
(a routine settle of 32 plus learning while aroused: 781 to 1,397 learning sweeps per
life against 2,674 to 3,481 for the restless walker without the copy, whose moments are
cheaper, 12 to 18 sweeps, because a held state settles fast).

### reward/3: need 0.9 on fresh seeds 501 to 505, gate failed on learned credit

[`protocol-reward-3.json`](protocol-reward-3.json), SHA-256
`7fc1bc9b5bf9d09a14b0b1b518c088499733296161fc1596977332d732f43967`: reward/2's point with
the need at 0.9, above the limp's income of 0.67, so a limping walker stays roused; reward/2's
gates. Receipt `results/confirmation-reward-3-2026-10-08.json.gz`, verified.

| seed | live | eligibility 0.95 (control) | defaults | frozen | nocopy | step | tabular | random |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 501 | 1.00 / 1.00 / 0.00 | 1.00 / 1.00 / 0.00 | 1.00 / 1.00 / 0.00 | **1.00** / 1.00 / 0.00 | 0.54 / 0.00 / 1.00 | 0.98 / 1.00 / 1.00 | 0.97 | 0.57 |
| 502 | **1.00** / 1.00 / 0.00 | 1.00 / 1.00 / 0.00 | 0.60 / 0.00 / 1.00 | 0.00 / 0.00 / 0.00 | 0.54 / 0.00 / 1.00 | 0.98 / 1.00 / 1.00 | 0.95 | 0.46 |
| 503 | **1.00** / 1.00 / 0.00 | 1.00 / 1.00 / 0.00 | 1.00 / 1.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.48 / 0.00 / 1.00 | 0.98 / 1.00 / 1.00 | 0.98 | 0.49 |
| 504 | **1.00** / 1.00 / 0.00 | 1.00 / 1.00 / 0.00 | 0.56 / 0.00 / 1.00 | 0.00 / 0.00 / 0.00 | 0.56 / 0.00 / 1.00 | 1.00 / 1.00 / 1.00 | 0.94 | 0.54 |
| 505 | 1.00 / 1.00 / 0.00 | 0.59 / 0.00 / 1.00 | 1.00 / 1.00 / 0.00 | **1.00** / 1.00 / 0.00 | 0.49 / 0.00 / 1.00 | 0.98 / 1.00 / 1.00 | 0.95 | 0.43 |

All five live walkers alternate at 1.00, calm, with their greedy probes at 1.00 and their twins
identical, and no founder limps; the copyless walker is aroused at every moment of the window,
as its income remains below the need of 0.9. Two founders, 501 and 505, had the
beat from birth (frozen 1.00), so the gate on learned beats credits 3 of 5 and fails as declared.
The need above the limp's income removes the limp on these seeds; a rule that screens founders
by their frozen arm before the gate, declared up front, would be the next protocol. The
eligibility control at need 0.9 is restless on seed 505 (0.59) and the composed defaults on two.

### Historical receipts on Cadence 0.76.0

The three reward receipts in `results/` were regenerated on the released `cadence-net` 0.76.0
(the efference copy as merged, with the audit's checkpoint validation and boundary bias
exclusion); every founder's action sequence is identical to the receipt taken on the branch
before the release.

### Maintainer audit with instrument revision 2

The corrected chamber at `0a0a0d7` reran all three unchanged protocols, with 40
founders each and no cap or override. All 120 main-life action sequences, rewards,
arousal readings, scores and greedy probes match the historical receipts. The
same-founder gates remain failed / passed / failed, with 3/5, 4/5 and 3/5 jointly
qualifying founders respectively. The runtime differs from 0.76.0 only in its
version string; the corrections are in the instrument.

With the checkpoint's actual world boundary restored, recovery after each tested
pause or distractor is 3/5, 4/5 and 5/5 for the live arm of reward/1, reward/2 and
reward/3. The corresponding eligibility controls recover in 5/5, 4/5 and 4/5.
In reward/2, the corrected live recovery is 4/5; the historical instrument's 3/5
also included its unintended missing-reward disturbance. Initial/final saves,
greedy-probe time and baseline calls are now included in work accounting.

Separate receipts retain the corrected measurements and source manifests:
[`reward/1 audit`](results/audit-reward-1-maintainer-2026-10-08.json.gz),
[`reward/2 audit`](results/audit-reward-2-maintainer-2026-10-08.json.gz), and
[`reward/3 audit`](results/audit-reward-3-maintainer-2026-10-08.json.gz).
Their canonical forms, sources, artifacts, event arithmetic, census and gates
verify. These reuse the original seeds as an instrument audit, not fresh
confirmation. The historical tables above remain unchanged.

### Reading the reward chambers

The frozen arm is the zero of learning: a greedy founder under identical drive holds one
foot unless the copy's random projection happens to make a two-cycle, which it did on four
of fifteen fresh founders and none of six development founders; a founder with the beat from
birth is reported and not credited. The walker without the copy is the control for the
mechanism: it earns occasional rewards while exploring but its greedy probe holds one
foot. It stays restless at need 0.5 or 0.9, or calms into mostly holding one foot at
need 0 in development. The always-learning `step` loop shows that the
beat can be learned by sampling every moment; `live` shows that it can be learned in
youth and then walked as routine, with learning sweeps a third of the restless control's
and no aroused moment in the window. The tabular learner is what the one bit the copy
carries is worth to a learner that reads it directly.

The limp is the residual failure: a period-three attractor, `RLL`, that pays two steps in
three, above a need of 0.5, so the walker is calm in it and routine learns nothing. It is
the manifesto's internally coherent equilibrium that is wrong about the world, measured:
one founder in five in reward/2, two in five in reward/1. With need 0.9, reward/3
has no limp on its five fresh founders, but two already had the beat from birth.
Its learned-credit gate therefore still fails; these results do not establish a
generally better default.

### Failures and limits of the reward chambers

- reward/1 failed its declared gate (3/5 acquired, one from birth, two limps) and stands.
  reward/2 passed with 4/5 learned, one limp; the eligibility hypothesis it declared is
  not confirmed as the cause of the limp, since the control limps on another founder.
  reward/3 (need 0.9) has no limp and 5/5 at 1.00 but fails its learned-credit gate on two
  founders with the beat from birth; it stands as declared.
- The task is one bit of history at a delay of one. Longer delays, more actions and a
  pause the walker should wait through are not measured; after a pause it steps on.
- The operating point is the key-door nursery's, not the composed defaults, which reach
  4/5, 3/5 and 3/5 acquisitions in reward/1, reward/2 and reward/3 with the copy. No default changes.
- A life is 600 moments on one founder with one stream; recovery after disturbances is
  3/5 for the walker and 4/5 for the control, with the limping founder never recovering.
- The paced and host-load variants of the supervised chamber are not repeated here.

## loop/1: an engineered sensory-history comparison for issue 140, 2026-10-08

Issue [#140](https://github.com/muellerberndt/cadence/issues/140) reports a brain that
predicts the next event of a pattern well on held-out rows and collapses, when fed its own
output, to one trajectory that does not depend on the prime: on the C64 rows of 0.73.1 the
loop held forever. A brain that cannot count rows since its last onset, hearing its own
holds, has hold as its only consistent answer, and the majority fixed point seals itself.
[`loop_rhythm.py`](loop_rhythm.py) is the smallest version of that observation: one voice,
an onset every four rows, the heard event and a constant drive as the only input, the next
event as the only label, 256 taught rows. After teaching, every arm plays four closed loops
from primes at every phase of the pattern: 16 real rows, then 64 rows hearing nothing but
its own last answer. [`protocol-loop.json`](protocol-loop.json) declares the pattern, the
C64 composer's recipe of 0.73.1 at the rate of 0.01 it measured as the edge, the copy, the
teaching contract, the seeds and the gates.

The `copy` arm supplies a decaying history of heard inputs by directly editing the brain's
efference storage. This is an instrument-side adapter, not the public own-command
efference contract or a supported application recipe. The learned graph still settles its
answer, and no future label enters this history; the adapter supplies a designed temporal
feature (0.35, 0.23, 0.15, 0.10 of the onset cell at decay 0.65). The heard event enters the copy
before the brain answers a row, in teaching, in the open-loop watching and in the prime of a
play, and the brain's own command is then taken back out of it, so the copy carries the
heard stream alone, one update per row; in the closed loop the brain's own command is the
heard event and stays. Teaching uses an explicit label-based mismatch policy: a row answered
right teaches nothing, a wrong or refused answer is followed by one lesson on that row, with
the trace and the copy put back to what the act read. Arms on the same founder weights:
`copy` (the sensory-history adapter), `own` (the same copy written only by the brain's own greedy
commands, the simpler control of the contract), `nocopy` (the working trace alone, the
0.73.1 setting), `frozen` (the `copy` founder without lessons, played under the same
contract), `ngram` (a table over the last four events taught on the same rows, the matched
conventional learner that counts explicitly), `hold` (hold forever) and `random`. Readings
per founder: the open-loop watching (from reset, the taught rows once more with free greedy
acts and no lesson), the onset rate of the plays against the truth's, agreement with the
primed pattern's own continuation, agreement with the other phases' continuations (prime
dependence: a play must agree with its own prime's continuation more than with every other
phase's) and the pairwise correlation of the plays across primes, the issue's own measure.
The gate, declared before the fresh seeds ran: at least four of five `copy` founders fire
within 20 percent of the truth's onset rate, agree with their primes' continuations at least
0.05 above hold forever (0.75), are prime dependent at every phase, and are not already
above that agreement untaught; no refused act.

That gate evaluates the declared adapter comparison. It does not establish a repair of
#140 in the native continuing brain: the public own-command and trace-only arms fail here.
The conditional teacher policy also differs from `Brain.live`, whose arousal law can learn
from correct outcomes during youth or sustained arousal. The n-gram receives the same
event stream with explicit history; memory representation and its cost differ between arms.

```sh
python benchmarks/rhythm/loop_rhythm.py --out /tmp/loop-1
python benchmarks/rhythm/loop_rhythm.py --verify /tmp/loop-1
```

The frozen run, seven arms on the five confirmation seeds, takes about two minutes on one
laptop CPU. Changed seed, arm, pass or protocol selections mark a receipt not frozen.

The tables and original receipts below retain their original source and metrics. Maintainer
instrument revision 2 records raw act/lesson events, checks scores against those events,
requires every planned control and each prime's threshold, and counts refused answers as
misses. Missing same-founder frozen controls cannot earn learning credit; refusals during
teaching, watching, primes or continuation also prevent acceptance. The instrument records
adapter/control work and checkpoint I/O separately and tests a mid-loop save/resume branch.
Its receipts identify the adapter explicitly; no library behavior or frozen protocol changed.
Corrected reruns use separate receipts and already-used seeds are audits, not fresh confirmation.
The historical receipts' original verification is not a revision-2 arithmetic or continuation
check; reproducing that verification requires their original source snapshot. The current
verifier requires revision-2 event records and checkpoint artifacts.

The original first-contract and every-row development receipts contain 10 and 11 refused
acts, respectively. Their teaching/watching metrics dropped refused rows from denominators,
and their play scores credited a refusal as HOLD. Interpret those historical comparisons
with that limitation; the old receipts lack the per-row teaching/watch events needed to
reconstruct every corrected metric. The original confirmation records zero refused acts.

### Development, seeds 0 to 5

The first contract issued the copy after the act during a play's prime, so the copy ran one
row behind the heard stream there and nowhere else; 2 of 6 founders looped
(`results/development-loop-first-contract-2026-10-08.json.gz`). With the contract corrected
and the open-loop watching added, the lesson-on-every-row recipe of the C64 composer showed
the real failure: the pattern came and went from pass to pass. Founder 0 at decay 0.8 read
0.75, 0.86, 0.97, 0.99, 0.75, 0.97, 1.0, 0.9, 0.76, 0.99 and ended at 0.75 after 24 passes
of 255 lessons each; which founders watched the pattern at the end was a matter of when
teaching stopped. With lessons on surprise only, every founder learned within the first
passes, the lessons stopped, and the loop followed. Founders watching the pattern at 0.99
or above after teaching, founders that learned the loop under the gate, and the lessons
given per founder (`results/development-loop-*.json.gz`):

| lessons | copy decay | passes | watched ≥ 0.99 | learned the loop | lessons per founder |
| --- | --- | --- | --- | --- | --- |
| every row | 0.5 | 24 | 0 of 6 | 0 of 6 | 6,120 |
| every row | 0.65 | 24 | 3 of 6 | 3 of 6 | 6,120 |
| every row | 0.8 | 24 | 0 of 6 (one at 0.96) | 0 of 6 | 6,120 |
| every row | 0.9 | 24 | 2 of 6 | 0 of 6 | 6,120 |
| every row | 0.5 | 48 | 1 of 6 | 1 of 6 | 12,240 |
| every row | 0.8 | 48 | 0 of 6 | 0 of 6 | 12,240 |
| on surprise | 0.5 | 24 | 6 of 6 | 5 of 6 | 5 to 96 |
| on surprise | **0.65** | 24 | 6 of 6 | 5 of 6 | 9 to 145 |
| on surprise | 0.8 | 24 | 5 of 6 (one at 0.93) | 5 of 6 | 50 to 409 |

Decay 0.65 had the highest mean agreement of the loops (0.980 against 0.959 and 0.928) and
was declared. The residual failure under the gate is one founder at one phase: founder 3
at decay 0.65 watches the pattern at 1.00 and loops it from three of the four primes, but
from the prime that ends on an onset its first onset comes one row late and the loop keeps
that phase for all 64 rows (0.52 against its own prime's continuation, 0.98 against the
phase-0 continuation), so it is not prime dependent at every phase. The same founder slips
at two phases at decay 0.5; founder 4 at decay 0.8 watches at 0.93 and limps between gaps
of three and four rows. The every-row founders that watched the pattern at 0.98 or better
at decay 0.9 looped it at 0.88 and 0.90 without prime dependence: at that decay the count
is too flat to hold the phase.

### Historical confirmation on seeds 601 to 605, adapter gate passed

Protocol SHA-256 `13499a1dae5861e51967bd8f3bde08c1d6edbb075bf996409aab2a18f8a81023`; receipt
`results/confirmation-loop-2026-10-08.json.gz`, verified, 35 founder-arms, 124 seconds, no
refused act or lesson. Watching is the open-loop agreement after teaching; loop is the mean
agreement of the four plays with their primes' own continuations; lessons are the lessons
given in 24 passes of 255 rows.

| seed | copy watching / loop / prime dependent / lessons | own watching / loop / lessons | nocopy watching / loop / lessons | frozen watching / loop | ngram | hold | random |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 601 | **1.00 / 1.00 / yes / 25** | 0.75 / 0.61 / 2,263 | 0.75 / 0.72 / 3,028 | 0.25 / 0.25 | 1.00 | 0.75 | 0.46 |
| 602 | **1.00 / 1.00 / yes / 28** | 0.75 / 0.75 / 2,272 | 0.75 / 0.67 / 3,015 | 0.25 / 0.41 | 1.00 | 0.75 | 0.46 |
| 603 | **1.00 / 1.00 / yes / 7** | 0.25 / 0.25 / 2,253 | 0.74 / 0.75 / 3,023 | 0.75 / 0.75 | 1.00 | 0.75 | 0.54 |
| 604 | **1.00 / 1.00 / yes / 42** | 0.25 / 0.25 / 2,262 | 0.73 / 0.73 / 3,026 | 0.25 / 0.38 | 1.00 | 0.75 | 0.47 |
| 605 | **1.00 / 1.00 / yes / 126** | 0.75 / 0.68 / 2,368 | 0.75 / 0.75 / 3,021 | 0.49 / 0.63 | 1.00 | 0.75 | 0.45 |

Five of five founders using the sensory-history adapter learned the loop: every play at every phase agrees with its own
prime's continuation at 1.00, fires at the truth's rate and keeps the prime's phase, after 7
to 126 lessons; the pairwise correlation of their four plays is −0.33, the correlation of
the four phase shifts of a period-four pattern, as for the n-gram table. The untaught
founders fire on nearly every row (601, 602, 604), hold (603) or mix (605): none had the
loop from birth. Original work: each taught brain arm made 6,691 acts (the untaught arm
made 571); `copy` used 32 sweeps per act (214,112 sweeps total for
`copy`, 153,408 to 164,032 for `nocopy`, whose brain is smaller); the `copy` lessons cost
122 to 1,306 sweeps against 17,000 to 30,000 for the arms that never learned.

### Corrected audit and native-memory development

The separate [revision-2 audit](results/audit-loop-revision2-2026-10-08.json.gz),
source `8e508141`, repeats all 35 original cells on the already used seeds.
Every adapter founder still meets the gate at every prime; all 20 brain lives
have identical saved continuations and no refused act. Raw event arithmetic,
complete controls, work and artifact custody verify. This is a corrected audit,
not another fresh confirmation, and the adapter's information remains explicit.

A bounded native comparison then changed only working-trace amplitude/decay:
historical `(0.1, 0.95)`, shorter `(0.3, 0.8)`, and composed `(3.0, 0.2)`, each
with native own-command copying or no copy, on development seeds 0 and 1.
All 12 taught lives and their 12 matched untaught lives are retained under
`results/development-native-*.json.gz`; no external sensory-history writes occur.
The no-copy brain on seed 0 plays every prime perfectly at both alternative
trace settings and beats its untaught twin. Seed 1 fails at both settings;
the historical trace and every own-command variant fail the complete behavior
screen. There are no refused acts. This demonstrates native learned generation
on one development founder, while reliability, fresh confirmation and the C64
corpus remain unestablished.

The predeclared follow-up kept the composed trace and no copy, adding development
seeds 2–5 with matched untaught, n-gram, hold and random controls. None of these
four learned the correct phase at every prime; aggregate taught agreement ranges
from 0.656 to 0.750 against 0.250–0.258 untaught. All n-gram controls pass; hold
and random fail. The complete result is **1/6 development founders**, below the
5/6 threshold declared before extending the sample, so no fresh confirmation
runs. Both [taught](results/development-native-extension-taught-2026-10-08.json.gz)
and [untaught](results/development-native-extension-untaught-2026-10-08.json.gz)
receipts verify from the retained `8e508141` source; no act was refused.
**Issue 140 remains open; no default is promoted.**

### Native sensory history, bounded follow-up of 2026-10-09

[`native_history.py`](native_history.py) preserves a successful opt-in configuration
of the existing public `Trace`. It copies settled **sensory** activation after each
issued action into the first three coordinates of the original 32-wide prefrontal
region. The next answer reads the preceding history. The graph, initial parameters,
projection scale and local learning law match the association-trace control;
the remaining 29 prefrontal coordinates receive no trace drive. Amplitude 3.0 and
decay 0.2 are retained. No observed event is inserted before its answer, and teacher
labels never write the trace. This differs from the historical heard-event adapter.

With the original mismatch-only teacher, 24 passes of 255 actions, and no supplied
clock, all six development founders and the five subsequently admitted founders
1601–1605 reproduce all four primed continuations exactly. The admission rule was
at least five of six development founders before confirmation, then four of five
confirmation founders. Every acquired founder beats its matched untaught control;
erasing only the two event-history coordinates after each act, while retaining the
constant-history coordinate, breaks the result. Restoring the checkpoint restores
the outputs. Acquired pending-action continuation and private-imagination isolation
also pass. No act or lesson refused, and no teaching phase exhausted its budget.
Teaching uses finite phases (`qualified=False`); their recorded residuals are
not equilibrium certificates. Qualified settlement applies to the issued free
answers.

| Original research stage | Founders passing every prime | Agreement at every prime | Onsets per row |
| --- | --- | --- | --- |
| Development, seeds 0–5 | 6/6 | 1.00 | 0.25 |
| Confirmation, seeds 1601–1605 | 5/5 | 1.00 | 0.25 |

The [investigation bundle](results/native-history-investigation-2026-10-09.json.gz)
retains the original research sources, protocols, results and preceding failed
comparisons, including numerical and cross-context audits, on runtime `7cd45bd`.
It is separate from receipts generated by this standalone instrument. Its
[fixed protocol](protocol-native-history.json) also retains the original
association-trace learner, untaught founder, event-only lesion, hold and uniform
random controls. Failures in the original association recipe are reported
independently and do not veto a successful sensory candidate. The random
expectation is 0.5; hold scores 0.75. This is component evidence for one voice at
period four, measured in event rows. It does not demonstrate an internal beat,
physical tempo or phrase-level organization. Each action settles the present
state with held history; future phrase variables are not jointly solved. The
reported multi-voice C64 case, broader sequence generation and issue 140 remain
unresolved; no default changes.

The [standalone reproduction receipt](results/native-history-reproduction-2026-10-09.json.gz)
repeats the same 6/6 development and 5/5 confirmation successes under this instrument,
with all 44 continuations exact and every preservation check passing. Its
[manifest](results/native-history-reproduction-2026-10-09.manifest.json) binds the
unmodified receipt, source hashes and execution details. The 66 checkpoint files
remain in the local run directory recorded there; complete checkpoint verification
requires those files. This repetition is an audit, not new fresh confirmation.

Run into a new directory; full mode enforces development admission before running
the recorded confirmation seeds. Repeating those seeds is an audit of the original
confirmation, not a new fresh confirmation:

```bash
python benchmarks/rhythm/native_history.py --screen --out /tmp/native-history-screen
python benchmarks/rhythm/native_history.py --full --out /tmp/native-history-full
python benchmarks/rhythm/native_history.py --verify /tmp/native-history-full
```

### C64 follow-up of 2026-10-09: numerical admission is not composer repair

The [C64 follow-up summary](results/c64-followup-2026-10-09.json.gz) and
[manifest](results/c64-followup-2026-10-09.manifest.json) retain the numerical
diagnosis and failed transfer comparisons on runtime `7cd45bd` (0.79.0).
This is a source-bound summary with disclosed omissions: the original corpus,
raw arrays and binary checkpoints remain local. It is separate from the exact
synthetic period-four receipts above.

The completed C64 comparison uses one founder (seed 0), one 256-row source,
three voice streams, 48 teaching passes without pass resets, and 16 prime offsets.
Each prime supplies 16 heard rows before 64 generated rows. The native sensory
candidate uses 72 history coordinates, eight more than the historical 64-wide
prefrontal region; this is not a source-only change to the historical graph.
No six-source or fresh-founder confirmation is claimed.

| Candidate | True-stream event agreement | Mean generated event agreement | Exhausted teaching phases | Primes passing all gates |
| --- | --- | --- | --- | --- |
| Finite teaching, free budget 1024 / nudged 50, decay 0.2 | 0.751634 | 0.139648 | 307 | 0/16 |
| Qualified teaching, both budgets 1024, decay 0.2 | 0.832680 | 0.457682 | 0 | 0/16 |
| Qualified teaching, both budgets 1024, decay 0.95 | 0.614379 | 0.258789 | 0 | 0/16 |

Hold forever scores 0.609375 generated event agreement. All three completed arms
preserve exact saved continuation and issue no refused action or lesson. All three
original 200-sweep arms refused and remain in the evidence. Replaying one
frozen refused action with budget 1024 qualifies after 224 sweeps and gives the
correct three events. The original refusal preserved every serialized array.
Increasing the available numerical work addresses that refusal; qualified
training still fails the behavioral gates.

In the qualified decay-0.2 arm, mean onset rate is 1.155273 across three voices,
near the reference 1.171875, but mean onset F1 is only 0.564830 and event identity
is below hold. The diagnostic finds contexts requiring different actions whose
present inputs match and whose short sensory histories yield identical settled
activity. Increasing decay to 0.95 worsens acquisition and generation, so it is
retained as a rejected comparison. This supports investigating retained temporal
organization, not promoting a trace or solver-budget default. The later
corrective-experience comparison was prepared but never executed.

Issue 140 remains open. These measurements establish neither an internal beat
nor phrase organization, and they do not replace the existing #121/#116 contracts
or the [governing principle](https://github.com/muellerberndt/cadence/issues/109#coherent-behavior-over-time).

### Reading the loop controls

- `own`, the copy written only by the brain's own commands, is the contract's control: while
  the brain emits holds its copy never carries an onset, so the count it would need is not in
  its input; two founders fire on every row, three hold or nearly hold, and all five keep
  taking about 95 lessons a pass to the end.
- `nocopy`, the working trace alone, is the 0.73.1 setting: every founder holds, at 0.67 to
  0.75 against hold forever's 0.75, after about 125 lessons a pass to the end. The trace at
  amplitude 0.1 carries the settled state, not the count.
- `ngram` is what the four events of history are worth to a learner that reads them as a
  table: 1.00 everywhere. `hold` is the 0.75 any silent brain earns on a quarter-onset
  pattern; `random` sits at 0.45 to 0.54.
- `frozen` is the zero of learning under the same contract.

### Failures and limits of loop/1

- One of six development founders fails the loop at one phase at the declared decay, with a
  late first onset the loop then keeps; the fresh seeds had no such founder. The gate is
  four of five and passed at five.
- One voice, period four, 16-row primes. The C64 rows of the issue (several voices, longer
  periods) are not measured; the copy's count fades to 0.02 of the onset cell after eight
  rows at decay 0.65, so periods well beyond eight are outside its range at this amplitude.
- Lessons on surprise only is this chamber's declared contract, selected on the development
  seeds against the every-row recipe; it is not a library default and the rhythm, reward and
  key-door chambers keep their own contracts. The teacher compares each free answer with its
  supplied next-event label; this is not the arousal-based policy of `Brain.live`.
- The cross-prime correlation reported by the issue is read on plays that are already scored
  by agreement; a loop that is prime dependent and wrong would score low on both.


## Predeclared event-time completion of #116

[`protocol-timing.json`](protocol-timing.json) reuses the native `rhythm/2`
`efference` recipe unchanged. It adds an untaught matched founder, ordered and
shuffled schedules with the same interval multiset, unrounded due/begin/end/deadline
records, and physically paced numerical-budget and tolerance comparisons. Neither
time nor the action index enters the observations. `every` and `mismatch` are
explicit application teacher policies; this chamber does not exercise `Brain.live`.
The primary timing arm is `efference/every`; all four recipe/teacher combinations
receive the learning, history, disturbance and continuation comparisons.

Before inspecting development seeds 0–1, the protocol declares one joint threshold:
at least four of five fresh founders 1701–1705 must satisfy **all** learning, control,
recovery, refusal, timing and continuation criteria. Each primary row must alternate
at least 95% of pairs over 64 cue-free events, beat its untaught and random baselines
by 0.20, and retain at least 90% in every eight-event block. Each disturbance must
recover within eight events and retain 95% alternation over the final 24 events.
Every accepted cadence must preserve every reference action, issue no refused
act, finish by the next distinct due time, begin within half that deadline interval,
and keep issued-action phase drift within half the smallest positive interval and
issued-action period error within 25ms. Both use solve-end timestamps; request lateness
uses solve-start timestamps. The 16-sweep budget remains a recorded stress case outside the declared
supported envelope of 64 sweeps or more. Missing planned comparisons fail admission.

This is an externally scheduled, event-driven period-two trajectory. The clock
supplies due times; native brain history determines each action. It does not claim
an endogenous physical oscillator. Checkpoints include the world event cursor and
phase anchor, including the remaining pause events; physical forks reanchor their
monotonic host clock. The verifier independently recounts actions/refusals and solve
work, checks every saved continuation array and checkpoint-I/O record, and recomputes
the joint gate from raw actions and timestamps. The saved restored fork is reused,
removing the old instrument's duplicate uncharged replay.

[Development on seeds 0–1](results/development-timing-1.json.gz) completed all eight
declared recipe/teacher cases. Both
primary founders learned perfect alternation, beat their untaught and random
controls, and passed history, recovery and continuation checks. **Joint timing
passes: 0/2.** They missed 52 and 26 deadlines across accepted cases; maximum issued
period errors were 348.67ms and 312.36ms. The 50ms schedule accumulated start delay
up to 526.60ms. Even the regular 100ms case had a 79.60ms act on seed 0, while seed 1
passed that case with a maximum act of 11.06ms. This is a failed host-run envelope,
not a successful timing claim.

The first attempt used NumPy's Apple Accelerate backend without declaring its
separate thread limit. The instrument now sets `VECLIB_MAXIMUM_THREADS=1` before
NumPy import and records per-act process CPU time beside wall time. A same-seed,
same-gate rerun can test this resource configuration; the omission is not proven
to have caused the failure. Other user workloads were present and were left alone.
[The same-gate rerun](results/development-timing-2.json.gz), with the Accelerate
limit and CPU records, also completed all eight cases and verified: **joint passes
0/2**, with 29 and 10 accepted deadline misses. Both primary founders again passed
learning, history, recovery and saved continuation. In the slowest 50ms-cadence
acts, seeds 0/1 spent 247.29/166.28ms wall time but only 5.49/6.39ms process CPU;
outer recording overhead was 0.03ms. This points to waiting or descheduling, not
hundreds of milliseconds of brain computation or CPU-consuming garbage collection;
it does not identify the underlying OS/backend cause. Both acts qualified in
32 sweeps. The thread setting alone did not establish the declared timing envelope.

The updated focused suite passes 33 tests. Both original full capsules verified,
and the new verifier also checks the preserved first attempt. No fresh confirmation
has run, and no numerical gate was relaxed. **#116 remains open:** the bounded
learning/control/custody work is present, but the predeclared physical timing
contract still fails development on this host. The older rhythm/1, rhythm/2 and
reward receipts above retain their original bytes and scope; this new verifier
does not retroactively certify their timing or work accounting.

### Physical timing of the recurrent control

The original flip-flop comparisons measured event sequences without physical pacing.
[`timing_control.py`](timing_control.py) completes that missing comparison using the
already-trained `efference/every` control checkpoints from development seeds 0–1.
Its separate declaration freezes the original inputs, checkpoints, source hashes,
software/thread setup and all eight distinct physical schedules per founder before
any control action. It does not retrain or run a brain, change the timing protocol,
or claim fresh confirmation. Numerical solver settings do not apply to this control;
their physical schedule is the same regular 100ms schedule measured here.

[All 16 control cases](results/development-timing-control-1.json.gz) passed the
unchanged physical bounds in 54.98 seconds, including host load and shuffled time.
The 512 policy calls (2,048 rows) had no failures or missed deadlines. Maximum start
lateness was 14.71ms, issued-action phase drift 16.40ms, and issued-period error
12.00ms against the 25ms bound. Every action was independently reconstructed from
the saved policy, and checkpoint state, complete work and the full case census
verified. Dense policy work was 28,672 multiply-accumulates, excluding feature
construction, argmax, memory traffic and verification; measured policy CPU/wall
time was 0.065/0.085 seconds. Source: `e71b5131936de6dd8d55ab23db215a55570c4dc8`.

The first verifier rejected five rounded display values because subtracting relative
timestamps crossed a rounding midpoint by a few floating-point bits. The unchanged
capsule passes the corrected independent verifier at
`e3ff063e3d48de2049769ff8e6745e058e3a9831`: it checks the declared display precision,
while every physical gate still uses the original unrounded timestamps. The focused
suite passes 22 tests. No timed rerun or timing-bound change was needed.

This supplies the missing competent physical recurrent control for #116's comparison
item. It leaves the brain's **0/2 joint timing result** unchanged. Sequential runs
do not identify the cause of earlier host stalls, and fresh brain confirmation and
issue closure remain unsupported. Replay requires the preserved full development
capsule and its recorded environment:

```sh
python benchmarks/rhythm/timing_control.py --prepare /path/to/timing-development-02 --out /tmp/rhythm-control
python benchmarks/rhythm/timing_control.py --run /tmp/rhythm-control
python benchmarks/rhythm/timing_control.py --verify /tmp/rhythm-control
```
