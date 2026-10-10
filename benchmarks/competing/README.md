# The competing-skill ring (#169)

This bounded instrument asks whether a policy that one continuing `Brain.compose` life
acquired in a nursery survives continued learning at its own actor step, when the world then
brings a competing skill, unavoidable burn, both or neither, while the acquired skill keeps
paying. It is a table-world reproduction attempt of the failure that
[issue 169](https://github.com/muellerberndt/cadence/issues/169) measured in the robot arena,
on the arena's own founder brain, with the controls the issue names (smaller actor steps, a
raised arousal threshold, the arena's ring stage, a frozen copy, uniform random). It
establishes no default and closes no gate. Read the
[world-model guide](../../docs/world-model.md) and
[routine and repair](../../docs/continuous.md#routine-and-repair-live) before interpreting it.

**Development readings, 2026-10-10, Cadence 0.80.0 source.** In this world the acquired skill
is **not lost**: at the gene step, learning from every outcome of ten thousand aroused
moments under the weak payoff and from thousands under the certain one, skill A keeps its
greedy accuracy in every ring, and a soft skill is hardened by continued learning wherever the
ring brings no competing skill. The drops in greedy accuracy belong to readings taken with the
working trace carried into the probe on a policy that has not hardened: under the weak payoff
the frozen copy reads 0.50 carried and 0.67 reset, the door's value, with unchanged weights.
The smaller actor steps retain no more than the gene step and hold the competing skill less.
The arena's loss is therefore not a generic property of the actor law at these rates; it needs
the arena's own dynamics, and its driving test needs a trace-reset reading beside the carried
one. Gates are declared only after development; `protocol.json` carries `"gates": null`.

## What runs

**World.** Sixteen inputs, four actions. Four nursery situations (skill A) are sparse
patterns of four active inputs. Four ring situations (skill B) each keep two inputs of a
nursery twin and take two others; the right actions of A are a permutation of the four,
those of B the same permutation shifted, so every twin asks a different action from a
half-overlapping input. The right action pays +1 with the declared contingency's probability
(`hard`: right 1.0, wrong 0.0; `soft`: right 0.6, wrong 0.3, the arena's weak contingency), a
burning moment adds −0.5 whatever the action. A competent nursery life earns 1.0 (hard) or
0.6 (soft) per moment.

**Brain.** The arena's founder (`cadence-robot-arena/arena/brain.py`, 2026-10-09): modules
(48, 24), sensory scale 4, learner temperature 0.3, actor eta 0.03 with eta_bias 0.003,
lam 0.6, gamma 0.95, eta_critic 5, working trace at amplitude 0.3 and decay 0.1, associative
memory at consolidation 0.05, founder arousal with youth 300. The arena's `need` is in its
own reward units; here it is 0.9 of the competent hard-nursery income, so a ring that pays
less keeps the life in want, the melee's regime. Everything else is the composed default.

**Phases.** A nursery of 4,000 moments on skill A, saved at the door of the ring with its
pending feedback. A ring of 10,000 moments. Four rings: `full` (A and B situations alike,
with burn), `competing` (A and B, no burn), `burn` (A only, with burn), `none` (A only, the
nursery continued). The situations and coins of a seed are identical across rings; burn
changes neither.

**Arms**, all continuing the same saved nursery brain of a seed, differing only in what
`retune` changes at the door: `gene` (nothing), `small` (actor eta 0.003), `smallest`
(0.001), `calm` (arousal threshold 0.5), `stage` (the arena's ring stage: eta 0.001, need 0,
heat 0, temperature 0.2, arousal references forgotten), `frozen` (greedy, no outcomes),
`random`.

**Readings.** Every 250 moments, on saved and reloaded copies so the living brain is never
read for a measurement: the greedy action for every situation, read twice, by a copy that
carries the living brain's working trace and warm state (`carried`, the arena's frozen
driving test) and by a copy whose stream state is reset first (`reset`: the same parameters,
no context); the base policy's probability of the right action and its margin over the best
other action. On the living brain, per window: aroused share, outcomes learned from, mean
dopamine and raw temporal-difference error, mean actor step, income. Since the door: the
RMS drift of the efficacies per projection and of the biases per region, relative to the
door's RMS. Over the ring: the sign consistency of each synapse's actor steps,
`|sum| / sqrt(n · sum of squares)`, one for a push that never changes sign, near zero for a
random walk.

## Results on the six development seeds

Receipts: `results/development-hard-2026-10-10.json.gz` and
`results/development-soft-2026-10-10.json.gz`, each 6 seeds × (5 arms × 4 rings + frozen +
random) = 132 ring lives on 10,000 moments, five to nine minutes on nine M4 cores.
`python benchmarks/competing/competing_skills.py --report <receipt>` prints the tables and
`--curves RING ARM` the skill-A accuracy along the ring per seed.

### Hard contingency: a certain payoff hardens the skill, and nothing loses it

At the door all six founders answer skill A at 1.00 with the right action at probability
0.77–0.83 (margin 0.70–0.77 over the best other action); the nursery was aroused 29–44 % of
its moments and learned from 1,146–1,763 outcomes.

| Ring | Arm | A at door | A at end, carried / reset | B at end, carried / reset | A p(correct) end | Aroused share | Learned | Drift assoc→motor | Consistency assoc→motor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | `gene` | 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 0.88 | 0.52 | 5,218 | 0.43 | 0.06 |
| full | `small` | 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 0.82 | 1.00 | 9,990 | 0.12 | 0.10 |
| full | `smallest` | 1.00 | 1.00 / 1.00 | 0.96 / 1.00 | 0.80 | 1.00 | 9,990 | 0.04 | 0.11 |
| full | `calm` | 1.00 | 1.00 / 1.00 | 0.92 / 0.96 | 0.77 | 0.08 | 787 | 0.07 | 0.07 |
| full | `stage` | 1.00 | 1.00 / 1.00 | 0.04 / 0.12 | 0.80 | 0.00 | 0 | 0.00 | – |
| full | `frozen` | 1.00 | 1.00 / 1.00 | 0.04 / 0.12 | 0.80 | – | – | – | – |
| competing | `gene` | 1.00 | 1.00 / 1.00 | 1.00 / 0.96 | 0.82 | 0.12 | 1,202 | 0.14 | 0.06 |
| burn | `gene` | 1.00 | 1.00 / 1.00 | 0.17 / 0.08 | 0.88 | 0.39 | 3,858 | 0.37 | 0.10 |
| none | `gene` | 1.00 | 1.00 / 1.00 | 0.04 / 0.12 | 0.80 | 0.00 | 0 | 0.00 | – |

Mean over seeds; every arm's minimum on skill A is 1.00 at the end, carried or reset, in every
ring (`--report` prints the full tables). The gene step moves the association→motor efficacies
by 0.43 of their RMS in the full ring and 0.37 under burn alone, with the learned moments'
step sign consistency at 0.06–0.10: the steps are noise and the weights random-walk, yet a
readout with margins of 0.68–0.84 keeps every greedy answer. A life in the `none` ring earns
its need and stays calm: nothing moves. `stage` is calm throughout and learns nothing; here it
is a frozen brain.

### Soft contingency: the arena's soft policy, kept and partly hardened

With the right action paid six times in ten and a wrong one three times in ten, the nursery
is in want throughout (income 0.38–0.40 against a need of 0.9) and learns from every one of
its 3,999 outcomes. At the door the founders answer skill A at 0.75 on average with the trace
carried (0.50 to 1.00) and at 0.67 with the trace reset, with the right action at probability
0.38–0.58 and margins of 0.01 to 0.39 on the reset reading: a soft policy, as the arena's
greedy margins were, and one whose greedy answers depend on the context the trace carries.

| Ring | Arm | A at door, carried / reset | A at end, carried / reset | B at end, carried / reset | A p(correct) end, reset | A margin end, reset | Aroused share | Learned | Drift assoc→motor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | `gene` | 0.75 / 0.67 | 0.71 / 0.67 | 0.58 / 0.54 | 0.57 | 0.29 | 1.00 | 10,000 | 0.27 |
| full | `small` | 0.75 / 0.67 | 0.71 / 0.75 | 0.46 / 0.42 | 0.52 | 0.27 | 1.00 | 10,000 | 0.03 |
| full | `smallest` | 0.75 / 0.67 | 0.71 / 0.67 | 0.38 / 0.50 | 0.51 | 0.26 | 1.00 | 10,000 | 0.01 |
| full | `calm` | 0.75 / 0.67 | 0.75 / 0.79 | 0.67 / 0.71 | 0.60 | 0.35 | 0.99 | 9,866 | 0.28 |
| full | `stage` | 0.75 / 0.67 | 0.54 / 0.62 | 0.25 / 0.21 | 0.47 | 0.19 | 0.01 | 118 | 0.00 |
| full | `frozen` | 0.75 / 0.67 | 0.50 / 0.67 | 0.21 / 0.17 | 0.48 | 0.20 | – | – | – |
| competing | `gene` | 0.75 / 0.67 | 0.71 / 0.71 | 0.67 / 0.58 | 0.60 | 0.33 | 1.00 | 10,000 | 0.33 |
| competing | `small` | 0.75 / 0.67 | 0.71 / 0.79 | 0.50 / 0.38 | 0.52 | 0.25 | 1.00 | 10,000 | 0.04 |
| burn | `gene` | 0.75 / 0.67 | 0.92 / 0.83 | 0.17 / 0.08 | 0.70 | 0.52 | 1.00 | 10,000 | 0.40 |
| burn | `small` | 0.75 / 0.67 | 0.71 / 0.71 | 0.12 / 0.17 | 0.48 | 0.18 | 1.00 | 10,000 | 0.04 |
| none | `gene` | 0.75 / 0.67 | 0.92 / 0.83 | 0.17 / 0.12 | 0.67 | 0.45 | 1.00 | 10,000 | 0.44 |
| none | `smallest` | 0.75 / 0.67 | 0.75 / 0.67 | 0.17 / 0.17 | 0.48 | 0.19 | 1.00 | 10,000 | 0.02 |

Mean over seeds; `--report` prints every arm with minima. Three readings. First, with the
context removed no arm but `stage` in the full ring (0.62) ends below the door's 0.67 on skill
A, and where the ring brings no competing skill the gene step hardens the soft skill: the right
action's probability on the reset reading rises from 0.38–0.58 at the door to 0.67–0.70 at the
end and its margin to 0.45–0.52. With skill B present the gene step keeps A at 0.67–0.71
while acquiring B to 0.54–0.58. Second, the smaller steps harden less (margins 0.15–0.27 at
the end) and hold skill B at 0.38–0.50; the arena's "nothing else moves". Third, the carried
readings move with the context: the frozen copy, whose weights are the door's, reads 0.50 with
the ring's trace carried and 0.67 with it reset, and the arms with soft readouts read carried
and reset values that differ by up to 0.13 in either direction, because a policy that has not
hardened answers from its context. A frozen greedy test of a soft policy therefore measures
the trace as much as the weights.

## What this does and does not establish

It establishes that on this brain, at the arena's genes, an acquired situation→action policy
is not lost to continued aroused learning at the gene step over ten thousand moments, with
or without a competing skill, with or without noisy unavoidable punishment, for a certain or
a weak payoff; that the smaller steps the arena adopted retain no more here and acquire far
less; and that a frozen greedy test is sensitive to the context the working trace carries.

It does not reproduce the arena's loss and does not explain it. The arena differs in ways
this table-world lacks: several motor slots with one heated at a time, continuous senses and
a body whose commands change what it next senses, progress pay that stops once the skill has
succeeded (a competent approach reaches the dummy and is paid nothing more), fights that
punish the approach the nursery paid for, and a driving test that scores behaviour over time
from a carried trace. Each of these can lose a greedy test score without a synapse being
"forgotten": the first by exploration perturbing one slot while the others settle, the third
and fourth by revising a skill the world stopped paying for, the last by context. The
manifesto asks to distinguish such revision from interference, and the next bounded delivery
for #169 is therefore in the arena, with this chamber's controls carried over: a sixty-fight
protocol whose driving test reads a trace-reset copy beside the carried one, a nursery-only
refresher arm, the sign of the dopamine on approach-eligible moments per fight, and a test
of whether approach is paid in the ring. No library mechanism is proposed from this
evidence; the two-timescale synapse of the issue's second hypothesis would address a drift
this chamber did not find.

## Run and verify

From the repository root with the source on the path:

```sh
PYTHONPATH=src python benchmarks/competing/competing_skills.py --ring 10000 --workers 9 \
    --out benchmarks/competing/results/development-hard-<date>.json.gz
PYTHONPATH=src python benchmarks/competing/competing_skills.py --ring 10000 --workers 9 \
    --contingency 0.6 0.3 --out benchmarks/competing/results/development-soft-<date>.json.gz
python benchmarks/competing/competing_skills.py --report <receipt>
python benchmarks/competing/competing_skills.py --report <receipt> --curves full gene
python benchmarks/competing/competing_skills.py --verify <receipt> [--current]
python -m pytest -q benchmarks/competing
```

A receipt is a `cadence.Receipt` bound to `competing_skills.py` and every module of the
library as they were when the run began; `--verify --current` also requires the present
sources. Nurseries are saved to `--checkpoints` (a temporary directory by default) so every
arm of a seed starts from the same bytes. Overrides of the phases, the probe interval and the
contingency are recorded in the receipt's `overrides`.
