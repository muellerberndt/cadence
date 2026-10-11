# Prerequisite-chain nursery

This bounded instrument asks whether one continuing `Brain.compose` life learns a chain of
actions whose pay comes at the end, and what makes it collapse onto one action. It is the
development instrument of [sequential tasks and prerequisite chains](../../docs/sequential-tasks.md),
built for the failure users report (a brain that learns to eat in one move and cannot
learn to gather wood, gather stone, make fire and warm up), on the contracts of
[issue 111](https://github.com/muellerberndt/cadence/issues/111) (credit across delays),
[issue 113](https://github.com/muellerberndt/cadence/issues/113) (drives) and
[issue 84](https://github.com/muellerberndt/cadence/issues/84) (the hidden chain state as a
memory task). It establishes no default and declares no gate before its first readings.
Read [routine and repair](../../docs/continuous.md#routine-and-repair-live) and the
[competing-skill ring](../competing/README.md), whose brain and receipts it shares, before
interpreting results.

## The world

A chain of `steps` actions (four in the protocol) must be taken in order; two further
actions do nothing. Sixteen inputs: one bit per stage (how many prerequisites the creature
holds), one cue that is always on, and three background bits fixed per seed. The chain's
actions and their order are a permutation per seed. In the `observed` mode the stage bit is
on; in the `hidden` mode the observation is the same at every stage, and the right action
depends on what the creature did before. A right action advances the stage, a wrong one
leaves it. Pay: `end`, +1 at the completion only; `shaped`, +1/steps at every correct step
and +1 at the completion; `lessons`, +1 at the completion of episodes that begin at a
random stage with the prerequisites granted, the backward lessons interleaved. The need in
the arousal law is 0.9 of what a life that always takes the right action earns per moment
under `end` pay (`need_share`).

## Arms and controls

The brain is the arena founder of the competing ring (two modules of 48 and 24, sensory
scale 4, temperature 0.3, actor step 0.03 with a bias step of 0.003, lam 0.6, gamma 0.95,
critic rate 5, trace 0.3 / 0.1, consolidation 0.05, youth 300). Four arms from birth:
`gene`, the founder; `small`, a tenth of the actor step; `taper`, the gene step for the
first half of life and a tenth after (`retune`); `homeo`, the readout's intrinsic plasticity
(`learning_homeostasis_rate` 0.1 toward 0.3). Four controls: `random`, uniform over every
action; `stationary`, uniform over the chain's actions, the memoryless ceiling of the hidden
mode; `frozen`, the newborn founder acting greedily; `tabular`, Q-learning over the stage it
observes (one state in the hidden mode; alpha 0.1, epsilon 0.1, gamma 0.95), the competent
learner with the same information.

## Readings

Every `probe_every` moments (2,000 of 20,000) a saved copy of the brain is loaded, its
stream state reset, and run from stage zero under `end` pay without learning for
`probe_moments` (1,000): completions per thousand moments of the greedy policy and of the
sampled policy at the learner's temperature; the greedy answer, the right action's
probability and its margin at every stage (observed mode); and the share of those moments
spent on the single most chosen action. The life itself is tallied per window: completions,
right-action share, income, aroused share, learned moments, the mean dopamine of the
learned moments and the share of them with a positive signal, and the action histogram.
Receipts bind the chamber and every library source; `--verify --current` checks a receipt
against the shipped sources, `--report` prints its tables. `--learning` and `--brain`
override the brain point and are recorded in the receipt.

```sh
PYTHONPATH=src python benchmarks/chain/prerequisite_chain.py --workers 9
PYTHONPATH=src python benchmarks/chain/prerequisite_chain.py --steps 2 --modes observed --pays end lessons
PYTHONPATH=src python benchmarks/chain/prerequisite_chain.py --verify benchmarks/chain/results/<receipt>.json.gz --current
```

## What this does and does not establish

The chamber measures one declared founder brain in one table world. It does not have a
body whose commands change what it next senses, a competing need, or a reward that stops
paying once the skill succeeds. A completion rate at the tabular learner's level with the
state observed establishes credit across the chain's steps at that length; the same with
the state hidden establishes a memory horizon the trace does not have on fresh founders
today. Gates are declared only after the first development readings. Development results
are recorded below as they arrive.

## Development readings, 2026-10-11 (receipt `results/development-2026-10-11.json.gz`)

The receipt binds the chamber file as committed and the library sources of commit
`c922553b428e`, the branch head it ran on; `--verify` checks it, and `--verify --current`
passes in a checkout of that commit.

Six seeds, a chain of four actions, 20,000 moments per life, probes every 2,000 from saved
copies; the table reads the last probe. Completions per thousand moments of the greedy and
the sampled policy from stage zero under pay at the end; the greedy answer's accuracy over
the four stages and the right action's probability on a reset copy (observed mode); the share
of the life's moments on its most chosen action; aroused share; learned moments of 20,000;
the share of learned moments with positive dopamine.

| mode | pay | arm | completions greedy / sampled | stage accuracy | p(right) | one action | aroused | learned | dopamine positive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| observed | end | `gene` | 42 / 77 | 0.67 | 0.41 | 0.20 | 1.00 | 19,999 | 0.31 |
| observed | end | `small` | 0 / 51 | 0.33 | 0.28 | 0.20 | 1.00 | 19,999 | 0.28 |
| observed | end | `taper` | 42 / 62 | 0.58 | 0.34 | 0.20 | 1.00 | 19,999 | 0.31 |
| observed | end | `homeo` | 0 / 48 | 0.17 | 0.14 | 0.17 | 1.00 | 19,999 | 0.41 |
| observed | shaped | `gene` | 204 / 108 | 0.92 | 0.47 | 0.29 | 0.39 | 7,792 | 0.28 |
| observed | shaped | `small` | 83 / 74 | 0.83 | 0.36 | 0.21 | 0.73 | 14,558 | 0.25 |
| observed | shaped | `taper` | 108 / 94 | 0.88 | 0.46 | 0.28 | 0.53 | 10,580 | 0.28 |
| observed | shaped | `homeo` | 102 / 66 | 0.21 | 0.20 | 0.26 | 0.36 | 7,145 | 0.35 |
| observed | lessons | `gene` | 33 / 63 | 0.54 | 0.43 | 0.27 | 0.89 | 17,875 | 0.31 |
| observed | lessons | `small` | 0 / 51 | 0.38 | 0.29 | 0.23 | 1.00 | 19,899 | 0.29 |
| observed | lessons | `taper` | 24 / 60 | 0.54 | 0.40 | 0.25 | 0.93 | 18,612 | 0.31 |
| observed | lessons | `homeo` | 0 / 37 | 0.21 | 0.11 | 0.18 | 0.99 | 19,857 | 0.30 |
| observed | any | `random` | 42 | – | – | 0.18 | – | – | – |
| observed | any | `stationary` | 62 | – | – | 0.27 | – | – | – |
| observed | any | `frozen` | 0 / 42 | 0.12 | 0.17 | 1.00 | – | – | – |
| observed | any | `tabular` | 250 / 229 | – | – | 0.25 | – | – | – |
| hidden | end | `gene` | 0 / 12 | 0.25 | 0.23 | 0.35 | 1.00 | 19,999 | 0.06 |
| hidden | shaped | `gene` | 0 / 15 | 0.25 | 0.22 | 0.34 | 1.00 | 19,999 | 0.18 |
| hidden | lessons | `gene` | 0 / 9 | 0.25 | 0.23 | 0.40 | 1.00 | 19,996 | 0.07 |
| hidden | any | `small`, `taper`, `homeo` | 0 / 7 to 33 | 0.00 to 0.25 | 0.02 to 0.23 | 0.17 to 0.40 | 1.00 | 19,988 to 19,999 | 0.07 to 0.24 |
| hidden | any | `tabular` | 0 / 5 | – | – | 1.00 | – | – | – |
| hidden | any | `stationary` | 62 | – | – | 0.27 | – | – | – |

Five readings. First, with the state observed and pay at every correct step, the founder
brain completes the chain at 204 per thousand moments against the tabular learner's 250,
answers 0.92 of the stages right, and calms down (aroused 0.39) once its income exceeds its
need; the smaller steps learn the same chain more slowly (83 and 108). Second, with pay at
the end only, the greedy policy completes at uniform random's rate (42) while the sampled
policy completes at 77 and the greedy answer is right at 0.67 of the stages: the policy has
learned the early steps and not the whole chain, and a greedy reading alone understates it.
Third, backward lessons as interleaved starting stages did not help in this world (33 at the
gene step): the completions the granted prerequisites bring meet the need, the brain calms
and learns less, and the probe from stage zero finds a weaker chain than under pay at the
end. Fourth, with the state hidden no arm beats the stationary ceiling (62) or uniform random
(42): the sampled policies complete 7 to 33 and the greedy ones 0, the tabular learner with
one state collapses onto one action, and the frozen newborn's greedy copy answers with one
action in every situation (one-action share 1.00). The chain's state must be in the
observation. Fifth, the readout's intrinsic plasticity at its development point (rate 0.1
toward 0.3) lowers every reading under reward learning here (stage accuracy 0.17 to 0.21):
a six-action readout whose right answer stands near 1 and whose others stand near 0 has a
mean activation near 0.17, and a pull toward 0.3 flattens it. The gene as declared is a
teaching-time mechanism; under reward its target would have to follow the slot's size, and
no such result exists. No collapse onto one action occurred in any learning arm with the
state observed (one-action share 0.17 to 0.29): a brain that is in want learns from every
moment here, and the collapse users report arrives under a different arousal regime, which
this chamber does not yet declare.
