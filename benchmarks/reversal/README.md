# Odour nursery

This bounded instrument asks whether one continuing `Brain.compose` life can acquire a
rule, adapt when the rule turns over unannounced, return when it turns back, and keep an
unrelated skill throughout, after short and after long experience of the first rule. It
is the reversal chamber of
[issue 88](https://github.com/muellerberndt/cadence/issues/88), roadmap row 05 of
[issue 109](https://github.com/muellerberndt/cadence/issues/109), and the first
behavioural evidence for the routine-and-repair loop of
[issue 122](https://github.com/muellerberndt/cadence/issues/122). It measures one declared
System 1 operating point with arousal against the same brain without it, the released
defaults, two ablations of the brain, frozen, replay and reset controls, a tabular
learner with the same information and uniform-random actions. It establishes no default.
Read [routine and repair](../../docs/continuous.md#routine-and-repair-live), the
[world-model guide](../../docs/world-model.md) and
[numerical contracts](../../docs/contracts.md) before interpreting results.

## What runs

An animal meets one of four odours per trial, drawn at random, and avoids (0) or
approaches (1). Approaching the sugar odour pays +1, approaching another odour costs 1,
avoiding pays nothing. Odours 0 and 1 are the reversal pair: the sugar sits at odour 0
under rule A and at odour 1 under rule B. Odours 2 and 3 are the stable pair, the
unrelated skill: odour 2 is always sugar and odour 3 never. A life is one stream without
resets: rule A for the exposure (100, 300, 1,000, 3,000 or 10,000 trials), rule B for 600
trials and rule A again for 600. Nothing announces a change.

The world and its payoff are those of the reports behind the issue
([50](https://github.com/muellerberndt/cadence/issues/50),
[69](https://github.com/muellerberndt/cadence/issues/69)): a two-odour T-maze as a
bandit. Those reports ran cadence-net 0.20 and 0.42 learners whose interfaces no longer
exist, with three seeds and a lag read on greedy choices held for 20 trials. The nursery
keeps their world, adds the stable pair, runs the current `Brain.compose`, reads the lag
on executed actions and the greedy choices on saved copies, and ends rule B after 600
trials, so a slower reversal is reported as none.

Every arm of a seed lives the same odour sequence:

| Arm | What it is |
| --- | --- |
| `live` | `Brain.compose(4, 2, modules=(32,))` at the protocol's operating point, with `ArousalConfig()` at its founders, through `Brain.live` |
| `step` | the same brain and operating point without arousal, through `step`: it samples and learns at every moment (the simpler control) |
| `defaults` | `Brain.compose(4, 2, modules=(32,))` at the released defaults, through `step` |
| `memory-only` | the `live` brain with its actor's rates at zero: associative memory and critic learn, the graph's policy does not |
| `graph-only` | the `live` brain without its associative memory: the graph's reward learning alone |
| `frozen` | the `live` brain after rule A, answering greedily and receiving no outcome |
| `replay` | the `live` brain after rule A, answering greedily and taking no outcome from the world, while its own witnessed records of rule A are presented to its memory again, one per trial: equal presentations on old evidence |
| `reset` | a newborn `live` brain at every rule change |
| `tabular` | epsilon-greedy tabular Q-learning with the same odour, action and reward; alpha 1.0 and epsilon 0.1, selected on the development seeds |
| `random` | uniform random actions |

The operating point of one continuing stream is working-trace amplitude 0.3,
consolidation 0.25 and an actor rate of 0.1 with a bias rate of 0.01; the composed
defaults are 3.0, 0.05, 1.0 and 0.05. Its selection on the development seeds is recorded
[below](#how-the-operating-point-and-the-founders-were-selected).

Readings per rule: the share of optimal executed actions in the last 100 trials; the lag,
the first trial from which the next 40 executed actions are at least 90% optimal; the
greedy choice and the policy's approach probability per odour of a saved and reloaded
copy every 25 trials; the probability of approaching each odour under the behaviour that
acted (the greedy choice with certainty in routine, heated sampling when aroused) and
under the base policy, read from the living brain's own settled state at every trial,
with the probability of each executed action; how often each odour was met and
approached; the first executed approach at the newly rewarded odour and the number of
approaches executed there until the greedy choice turned, read on a copy after each of
the first 20 of them, which separate too few contradicting witnesses from a failure to
revise after them; whether the stable pair stayed right at the probes; the share of
aroused moments; and the work of the life. The work ledger counts moments and settling
sweeps per mode, eligibility and feedback sweeps, probes with their sweeps and checkpoint
files, reads and writes of the associative memory, replay presentations, brains built and
the sweeps of a refused attempt, and records the wall time of a moment in each mode. The
probe copies are read and the living brain is not: a life gives the same executed actions
with and without its probes, and reading a probability from the settled state changes
nothing. A refused answer raises; the life is recorded as crashed with the work it had
done.

## Gates, fixed before the confirmation run

Over the confirmation lives of the `live` arm at exposures of 300 and more, at least 90%
end each rule with 90% of their last 100 executed actions optimal, keep the stable pair
right on 95% of their probes, and spend no more than 20% of the second half of each rule
aroused; at each of those exposures the median reversal lag is at most 150 trials, a life
that never reverses counting as beyond it. Exposure 100 ends inside the youth and is
reported without a gate. [protocol.json](protocol.json) holds the gates, the seeds and
every setting. It was committed and pushed before its confirmation seeds were run. It is
the chamber's third freeze; [the first two](#the-earlier-freezes) are recorded below.

## Run and verify

```sh
python benchmarks/reversal/odour_nursery.py --seeds confirmation --out /tmp/nursery.json.gz
python benchmarks/reversal/odour_nursery.py --verify /tmp/nursery.json.gz --current
python benchmarks/reversal/odour_nursery.py --report /tmp/nursery.json.gz
python -m pytest -q benchmarks/reversal
```

The first command runs the frozen protocol: ten arms, five exposures and ten
confirmation seeds, 500 lives, in about eight minutes on nine laptop cores. `--arms`,
`--seeds` and `--exposures` select a part. `--genes`, `--point`, `--reliability`,
`--payoff` and `--jitter` override the arousal genes, the operating point and the world,
and mark the receipt `frozen_protocol: false`; a `null` in `--point` leaves that setting
at its released default. The receipt is a `cadence.Receipt` bound to the chamber's source
and every module of the library. `--verify` checks its canonical form and digest, source
manifest integrity, a nonempty unique plan, the ranges and dimensions of the recorded
readings, one row for every planned life and the gates recomputed from the rows. New
receipts include the original protocol text and verify its hash and any claim of frozen
settings; older receipts compare frozen settings when their protocol source is available.
`--current` also requires the source manifest and the protocol hash of the files present.
These checks validate the recorded summaries; they cannot reconstruct unrecorded executed
actions or independently prove that a run occurred. `--report` prints the
tables below. `results/` keeps the receipts quoted here. The guards run short lives of
the `live` arm and its controls, the checkpoint continuation during a reversal, the
independence of a life from its probes, the gate arithmetic and the receipt's custody.

## Results on ten fresh seeds, 2026-10-06

Receipt: `results/confirmation-2026-10-06.json.gz`. Protocol SHA-256
`2f60a49656f101f7a2489a99b98c1fc724178dd2e29100f3285ac431b45eab2c`, frozen; seeds 500 to
509; cadence 0.74.0 at commit
[`45e48cf`](https://github.com/muellerberndt/cadence/commit/45e48cf), NumPy 2.5.3,
Python 3.13.0, macOS arm64. All 500 lives completed and none refused an answer. **The
gates passed.** Of the 40 gated lives of the `live` arm, 40 acquired rule A, 39 reversed,
39 returned, 40 kept the stable pair and 40 returned to routine; the median reversal lags
were 15.5, 24.5, 30 and 17 trials at exposures of 300, 1,000, 3,000 and 10,000. One life,
at exposure 10,000, never approached the moved sugar: aroused for 16% of rule B, it
approached that odour with a behaviour probability of 0.017 over the rule and ended both
later rules avoiding the reversal pair.

### Rule A: optimal share of the last 100 actions, mean (minimum)

| Arm | 100 | 300 | 1,000 | 3,000 | 10,000 |
| --- | --- | --- | --- | --- | --- |
| `live` | 0.76 (0.64) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (0.98) |
| `step` | 0.76 (0.64) | 0.85 (0.76) | 0.97 (0.90) | 0.93 (0.43) | 0.98 (0.96) |
| `defaults` | 0.53 (0.45) | 0.53 (0.46) | 0.49 (0.45) | 0.51 (0.47) | 0.48 (0.41) |
| `memory-only` | 0.75 (0.63) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) |
| `graph-only` | 0.52 (0.46) | 0.60 (0.49) | 0.63 (0.45) | 0.75 (0.50) | 0.80 (0.45) |
| `frozen` | 0.76 (0.64) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (0.98) |
| `replay` | 0.76 (0.64) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (0.98) |
| `reset` | 0.76 (0.64) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (0.98) |
| `tabular` | 0.64 (0.42) | 0.91 (0.68) | 0.96 (0.92) | 0.95 (0.92) | 0.95 (0.92) |
| `random` | 0.49 (0.40) | 0.49 (0.43) | 0.49 (0.44) | 0.52 (0.43) | 0.49 (0.42) |

### Rule B: optimal share of the last 100 actions, mean (minimum)

| Arm | 100 | 300 | 1,000 | 3,000 | 10,000 |
| --- | --- | --- | --- | --- | --- |
| `live` | 0.97 (0.71) | 1.00 (0.99) | 1.00 (1.00) | 1.00 (1.00) | 0.97 (0.67) |
| `step` | 0.95 (0.86) | 0.97 (0.91) | 0.95 (0.79) | 0.87 (0.71) | 0.75 (0.47) |
| `defaults` | 0.50 (0.42) | 0.50 (0.42) | 0.51 (0.39) | 0.51 (0.42) | 0.51 (0.45) |
| `memory-only` | 0.97 (0.71) | 1.00 (0.98) | 1.00 (1.00) | 0.99 (0.95) | 1.00 (1.00) |
| `graph-only` | 0.54 (0.39) | 0.56 (0.41) | 0.56 (0.46) | 0.58 (0.42) | 0.55 (0.49) |
| `frozen` | 0.49 (0.42) | 0.51 (0.43) | 0.50 (0.42) | 0.52 (0.46) | 0.47 (0.39) |
| `replay` | 0.49 (0.42) | 0.51 (0.43) | 0.50 (0.42) | 0.52 (0.46) | 0.47 (0.39) |
| `reset` | 1.00 (1.00) | 1.00 (0.97) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) |
| `tabular` | 0.96 (0.92) | 0.95 (0.92) | 0.94 (0.88) | 0.95 (0.92) | 0.95 (0.93) |
| `random` | 0.48 (0.38) | 0.48 (0.40) | 0.51 (0.44) | 0.51 (0.45) | 0.50 (0.41) |

### Rule A again: optimal share of the last 100 actions, mean (minimum)

| Arm | 100 | 300 | 1,000 | 3,000 | 10,000 |
| --- | --- | --- | --- | --- | --- |
| `live` | 0.97 (0.72) | 1.00 (0.99) | 1.00 (1.00) | 1.00 (1.00) | 0.98 (0.76) |
| `step` | 0.96 (0.78) | 0.96 (0.78) | 0.98 (0.96) | 0.84 (0.37) | 0.82 (0.46) |
| `defaults` | 0.51 (0.45) | 0.49 (0.34) | 0.50 (0.45) | 0.53 (0.45) | 0.51 (0.44) |
| `memory-only` | 0.97 (0.72) | 1.00 (0.99) | 1.00 (1.00) | 0.99 (0.94) | 1.00 (1.00) |
| `graph-only` | 0.61 (0.49) | 0.59 (0.49) | 0.54 (0.38) | 0.62 (0.45) | 0.68 (0.46) |
| `frozen` | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) |
| `replay` | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) |
| `reset` | 1.00 (1.00) | 1.00 (0.97) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) |
| `tabular` | 0.96 (0.93) | 0.95 (0.93) | 0.94 (0.92) | 0.96 (0.91) | 0.95 (0.93) |
| `random` | 0.50 (0.37) | 0.52 (0.41) | 0.49 (0.41) | 0.52 (0.45) | 0.49 (0.43) |

### Reversal lag in trials, median (lives that reversed / lives)

| Arm | 100 | 300 | 1,000 | 3,000 | 10,000 |
| --- | --- | --- | --- | --- | --- |
| `live` | 41 (10/10) | 15 (10/10) | 24 (10/10) | 30 (10/10) | 16 (9/10) |
| `step` | 74 (10/10) | 63 (10/10) | 139 (10/10) | 212 (7/10) | 168 (6/10) |
| `defaults` | none (0/10) | none (0/10) | none (0/10) | none (0/10) | none (0/10) |
| `memory-only` | 33 (10/10) | 14 (10/10) | 26 (10/10) | 31 (10/10) | 21 (10/10) |
| `graph-only` | none (0/10) | 391 (1/10) | none (0/10) | 387 (1/10) | none (0/10) |
| `frozen` | none (0/10) | none (0/10) | none (0/10) | none (0/10) | none (0/10) |
| `replay` | none (0/10) | none (0/10) | none (0/10) | none (0/10) | none (0/10) |
| `reset` | 71 (10/10) | 79 (10/10) | 85 (10/10) | 75 (10/10) | 69 (10/10) |
| `tabular` | 123 (10/10) | 57 (10/10) | 92 (10/10) | 53 (10/10) | 39 (10/10) |
| `random` | none (0/10) | none (0/10) | none (0/10) | none (0/10) | none (0/10) |

### Return lag in trials, median (lives that returned / lives)

| Arm | 100 | 300 | 1,000 | 3,000 | 10,000 |
| --- | --- | --- | --- | --- | --- |
| `live` | 20 (10/10) | 32 (10/10) | 15 (10/10) | 24 (10/10) | 19 (10/10) |
| `step` | 23 (10/10) | 61 (10/10) | 80 (10/10) | 353 (7/10) | 118 (6/10) |
| `defaults` | none (0/10) | none (0/10) | none (0/10) | none (0/10) | none (0/10) |
| `memory-only` | 17 (10/10) | 26 (10/10) | 19 (10/10) | 27 (10/10) | 21 (10/10) |
| `graph-only` | 353 (1/10) | 204 (2/10) | 28 (1/10) | 138 (2/10) | 35 (3/10) |
| `frozen` | 0 (10/10) | 0 (10/10) | 0 (10/10) | 0 (10/10) | 0 (10/10) |
| `replay` | 0 (10/10) | 0 (10/10) | 0 (10/10) | 0 (10/10) | 0 (10/10) |
| `reset` | 82 (10/10) | 70 (10/10) | 78 (10/10) | 76 (10/10) | 72 (10/10) |
| `tabular` | 51 (10/10) | 59 (10/10) | 40 (10/10) | 50 (10/10) | 21 (10/10) |
| `random` | none (0/10) | none (0/10) | none (0/10) | none (0/10) | none (0/10) |

### Greedy choices after the reversal: first of two consecutive probes with every odour right, median trial (lives / lives)

| Arm | 100 | 300 | 1,000 | 3,000 | 10,000 |
| --- | --- | --- | --- | --- | --- |
| `live` | 50 (9/10) | 25 (10/10) | 25 (10/10) | 50 (10/10) | 25 (9/10) |
| `step` | 25 (10/10) | 37 (10/10) | 150 (9/10) | 225 (5/10) | 350 (4/10) |
| `defaults` | 300 (3/10) | 100 (3/10) | 225 (1/10) | none (0/10) | none (0/10) |
| `memory-only` | 50 (9/10) | 25 (10/10) | 37 (10/10) | 50 (10/10) | 25 (10/10) |
| `graph-only` | none (0/10) | 400 (1/10) | none (0/10) | none (0/10) | none (0/10) |
| `frozen` | none (0/10) | none (0/10) | none (0/10) | none (0/10) | none (0/10) |
| `replay` | none (0/10) | none (0/10) | none (0/10) | none (0/10) | none (0/10) |
| `reset` | 25 (10/10) | 25 (10/10) | 25 (10/10) | 25 (10/10) | 25 (10/10) |
| `tabular` | 137 (10/10) | 75 (10/10) | 137 (10/10) | 62 (10/10) | 62 (10/10) |
| `random` | none (0/10) | none (0/10) | none (0/10) | none (0/10) | none (0/10) |

### First executed approach at the new sugar odour after the reversal, median trial (lives / lives)

| Arm | 100 | 300 | 1,000 | 3,000 | 10,000 |
| --- | --- | --- | --- | --- | --- |
| `live` | 35 (9/10) | 23 (10/10) | 22 (10/10) | 30 (10/10) | 12 (9/10) |
| `step` | 8 (10/10) | 27 (10/10) | 104 (9/10) | 383 (7/10) | 129 (7/10) |
| `defaults` | 4 (9/10) | 6 (7/10) | 5 (10/10) | 3 (9/10) | 3 (7/10) |
| `memory-only` | 37 (9/10) | 18 (10/10) | 24 (10/10) | 30 (10/10) | 18 (10/10) |
| `graph-only` | 4 (9/10) | 17 (9/10) | 11 (9/10) | 20 (8/10) | 15 (8/10) |
| `frozen` | none (0/10) | none (0/10) | none (0/10) | none (0/10) | none (0/10) |
| `replay` | none (0/10) | none (0/10) | none (0/10) | none (0/10) | none (0/10) |
| `reset` | 2 (10/10) | 2 (10/10) | 4 (10/10) | 2 (10/10) | 4 (10/10) |
| `tabular` | 77 (10/10) | 60 (10/10) | 119 (10/10) | 61 (10/10) | 49 (10/10) |
| `random` | 5 (10/10) | 5 (10/10) | 6 (10/10) | 3 (10/10) | 5 (10/10) |

### Policy's approach probability at the new sugar odour when the rule turns, median (minimum)

| Arm | 100 | 300 | 1,000 | 3,000 | 10,000 |
| --- | --- | --- | --- | --- | --- |
| `live` | 0.318 (0.153) | 0.321 (0.194) | 0.332 (0.115) | 0.319 (0.161) | 0.315 (0.120) |
| `step` | 0.318 (0.153) | 0.155 (0.036) | 0.023 (0.009) | 0.009 (0.008) | 0.006 (0.006) |
| `defaults` | 0.462 (0.004) | 0.494 (0.005) | 0.763 (0.004) | 0.995 (0.004) | 0.995 (0.004) |
| `memory-only` | 0.425 (0.165) | 0.415 (0.207) | 0.430 (0.125) | 0.435 (0.194) | 0.415 (0.290) |
| `graph-only` | 0.463 (0.278) | 0.444 (0.306) | 0.409 (0.192) | 0.387 (0.138) | 0.345 (0.027) |
| `frozen` | 0.318 (0.153) | 0.321 (0.194) | 0.332 (0.115) | 0.319 (0.161) | 0.315 (0.120) |
| `replay` | 0.318 (0.153) | 0.321 (0.194) | 0.332 (0.115) | 0.319 (0.161) | 0.315 (0.120) |
| `reset` | 0.520 (0.448) | 0.501 (0.458) | 0.501 (0.448) | 0.506 (0.458) | 0.521 (0.448) |
| `tabular` | 0.050 (0.050) | 0.050 (0.050) | 0.050 (0.050) | 0.050 (0.050) | 0.050 (0.050) |
| `random` | 0.500 (0.500) | 0.500 (0.500) | 0.500 (0.500) | 0.500 (0.500) | 0.500 (0.500) |

### Approaches executed there until the greedy choice turned, median (lives whose choice turned / lives)

| Arm | 100 | 300 | 1,000 | 3,000 | 10,000 |
| --- | --- | --- | --- | --- | --- |
| `live` | 1 (9/10) | 1 (10/10) | 1 (10/10) | 1 (10/10) | 1 (9/10) |
| `step` | 1 (10/10) | 1 (10/10) | 1 (9/10) | 1 (6/10) | 1 (7/10) |
| `defaults` | 2 (6/10) | 0 (6/10) | 0 (7/10) | 0 (6/10) | 0 (7/10) |
| `memory-only` | 1 (9/10) | 1 (10/10) | 1 (10/10) | 1 (10/10) | 1 (10/10) |
| `graph-only` | 1 (7/10) | 2 (7/10) | 1 (7/10) | 2 (7/10) | 3 (7/10) |
| `frozen` | none (0/10) | none (0/10) | none (0/10) | none (0/10) | none (0/10) |
| `replay` | none (0/10) | none (0/10) | none (0/10) | none (0/10) | none (0/10) |
| `reset` | 0 (10/10) | 0 (10/10) | 0 (10/10) | 0 (10/10) | 0 (10/10) |
| `tabular` | 1 (10/10) | 1 (10/10) | 1 (10/10) | 1 (10/10) | 1 (10/10) |
| `random` | none (0/10) | none (0/10) | none (0/10) | none (0/10) | none (0/10) |

### Lives whose greedy choice there never turned: approaches executed there under rule B, median (lives / lives)

| Arm | 100 | 300 | 1,000 | 3,000 | 10,000 |
| --- | --- | --- | --- | --- | --- |
| `live` | 0 (1/10) | none (0/10) | none (0/10) | none (0/10) | 0 (1/10) |
| `step` | none (0/10) | none (0/10) | 0 (1/10) | 0 (4/10) | 0 (3/10) |
| `defaults` | 1 (4/10) | 0 (4/10) | 1 (3/10) | 1 (4/10) | 0 (3/10) |
| `memory-only` | 0 (1/10) | none (0/10) | none (0/10) | none (0/10) | none (0/10) |
| `graph-only` | 2 (3/10) | 2 (3/10) | 2 (3/10) | 0 (3/10) | 0 (3/10) |
| `frozen` | 0 (10/10) | 0 (10/10) | 0 (10/10) | 0 (10/10) | 0 (10/10) |
| `replay` | 0 (10/10) | 0 (10/10) | 0 (10/10) | 0 (10/10) | 0 (10/10) |
| `reset` | none (0/10) | none (0/10) | none (0/10) | none (0/10) | none (0/10) |
| `tabular` | none (0/10) | none (0/10) | none (0/10) | none (0/10) | none (0/10) |
| `random` | 76 (10/10) | 72 (10/10) | 73 (10/10) | 79 (10/10) | 77 (10/10) |

### Behaviour under rule B: probability of approaching the new sugar odour, mean over its visits, median over lives (minimum)

| Arm | 100 | 300 | 1,000 | 3,000 | 10,000 |
| --- | --- | --- | --- | --- | --- |
| `live` | 0.941 (0.003) | 0.960 (0.870) | 0.952 (0.942) | 0.939 (0.749) | 0.978 (0.017) |
| `step` | 0.939 (0.871) | 0.909 (0.710) | 0.631 (0.010) | 0.161 (0.007) | 0.341 (0.006) |
| `defaults` | 0.577 (0.005) | 0.583 (0.004) | 0.746 (0.004) | 0.980 (0.004) | 0.996 (0.004) |
| `memory-only` | 0.940 (0.003) | 0.967 (0.913) | 0.956 (0.931) | 0.935 (0.746) | 0.975 (0.795) |
| `graph-only` | 0.759 (0.000) | 0.346 (0.000) | 0.572 (0.000) | 0.781 (0.000) | 0.195 (0.000) |
| `frozen` | 0.000 (0.000) | 0.000 (0.000) | 0.000 (0.000) | 0.000 (0.000) | 0.000 (0.000) |
| `replay` | 0.000 (0.000) | 0.000 (0.000) | 0.000 (0.000) | 0.000 (0.000) | 0.000 (0.000) |
| `reset` | 0.989 (0.984) | 0.987 (0.983) | 0.986 (0.981) | 0.986 (0.980) | 0.988 (0.984) |
| `tabular` | 0.836 (0.555) | 0.869 (0.461) | 0.784 (0.473) | 0.827 (0.534) | 0.876 (0.583) |
| `random` | 0.500 (0.500) | 0.500 (0.500) | 0.500 (0.500) | 0.500 (0.500) | 0.500 (0.500) |

### Stable pair right at the probes, mean (minimum)

| Arm | 100 | 300 | 1,000 | 3,000 | 10,000 |
| --- | --- | --- | --- | --- | --- |
| `live` | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) |
| `step` | 1.00 (1.00) | 1.00 (0.96) | 0.99 (0.75) | 0.96 (0.38) | 0.81 (0.00) |
| `defaults` | 0.06 (0.00) | 0.06 (0.00) | 0.03 (0.00) | 0.02 (0.00) | 0.01 (0.00) |
| `memory-only` | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) |
| `graph-only` | 0.12 (0.00) | 0.18 (0.00) | 0.19 (0.00) | 0.36 (0.00) | 0.46 (0.00) |
| `frozen` | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) |
| `replay` | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) |
| `reset` | 0.96 (0.96) | 0.97 (0.92) | 0.98 (0.96) | 0.97 (0.96) | 0.97 (0.92) |
| `tabular` | 0.96 (0.71) | 0.91 (0.12) | 0.98 (0.81) | 0.99 (0.94) | 1.00 (0.98) |
| `random` | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) |

### Work per life at the longest exposure, medians over lives

| Arm | answer sweeps per routine moment | per aroused moment | learning sweeps | probe sweeps | checkpoint files | memory reads | memory writes | presentations | brains | refused sweeps | routine ms (median, p90) | aroused ms (median, p90) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `live` | 32.0 | 20.9 | 1482 | 57568 | 2250 | 11364 | 163 | 0 | 1 | 0 | 0.50, 0.69 | 1.24, 1.62 |
| `step` | 0.0 | 25.9 | 138413 | 58608 | 2250 | 22401 | 11200 | 0 | 1 | 0 | none | 1.51, 2.15 |
| `defaults` | 0.0 | 5.5 | 59446 | 57600 | 2250 | 22401 | 11200 | 0 | 1 | 0 | none | 1.34, 1.99 |
| `memory-only` | 32.0 | 15.4 | 1336 | 57568 | 2250 | 11352 | 150 | 0 | 1 | 0 | 0.62, 0.92 | 1.52, 2.19 |
| `graph-only` | 32.0 | 24.1 | 10338 | 58736 | 2268 | 0 | 0 | 0 | 1 | 0 | 0.58, 1.12 | 1.41, 2.49 |
| `frozen` | 32.0 | 20.2 | 1120 | 57440 | 2245 | 11308 | 107 | 0 | 1 | 0 | 0.66, 1.01 | 1.61, 2.26 |
| `replay` | 32.0 | 20.2 | 1120 | 57440 | 2245 | 11308 | 1307 | 1200 | 1 | 0 | 0.57, 0.83 | 1.50, 2.00 |
| `reset` | 32.0 | 20.8 | 3232 | 57568 | 2250 | 11510 | 307 | 0 | 3 | 0 | 0.56, 0.80 | 1.28, 1.79 |
| `tabular` | 0.0 | 0.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | none | 0.00, 0.00 |
| `random` | 0.0 | 0.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | none | 0.00, 0.00 |

### The live arm: witnesses, arousal and work, medians over lives

| Exposure | first approach at the new sugar odour | from it to the turned greedy choice | aroused, whole life | aroused, second half of rule A | sweeps per routine moment | sweeps per aroused moment | learning sweeps per aroused moment |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 100 | 35 (9/10) | 0 (9/10) | 0.112 | 1.000 | 31.9 | 20.7 | 10.6 |
| 300 | 23 (10/10) | 0 (10/10) | 0.097 | 0.000 | 31.9 | 21.5 | 10.9 |
| 1,000 | 22 (10/10) | 0 (10/10) | 0.066 | 0.000 | 32.0 | 21.0 | 10.7 |
| 3,000 | 30 (10/10) | 0 (10/10) | 0.036 | 0.000 | 32.0 | 20.5 | 11.0 |
| 10,000 | 12 (9/10) | 0 (9/10) | 0.014 | 0.000 | 32.0 | 20.9 | 10.5 |

### Gates

```json
{
 "100": {
  "acquired": 0.0,
  "calm": 0.0,
  "crashed": 0,
  "lives": 10,
  "returned": 0.9,
  "reversed": 0.9,
  "stable_kept": 0.0
 },
 "1000": {
  "acquired": 1.0,
  "calm": 1.0,
  "crashed": 0,
  "lives": 10,
  "returned": 1.0,
  "reversed": 1.0,
  "stable_kept": 1.0
 },
 "10000": {
  "acquired": 1.0,
  "calm": 1.0,
  "crashed": 0,
  "lives": 10,
  "returned": 0.9,
  "reversed": 0.9,
  "stable_kept": 1.0
 },
 "300": {
  "acquired": 1.0,
  "calm": 1.0,
  "crashed": 0,
  "lives": 10,
  "returned": 1.0,
  "reversed": 1.0,
  "stable_kept": 1.0
 },
 "3000": {
  "acquired": 1.0,
  "calm": 1.0,
  "crashed": 0,
  "lives": 10,
  "returned": 1.0,
  "reversed": 1.0,
  "stable_kept": 1.0
 },
 "passed": true,
 "pooled": {
  "acquired": 1.0,
  "calm": 1.0,
  "crashed": 0,
  "lives": 40,
  "returned": 0.975,
  "reversed": 0.975,
  "stable_kept": 1.0
 },
 "reversal_lag": {
  "1000": 24.5,
  "10000": 17.0,
  "300": 15.5,
  "3000": 30.0
 }
}
```

### Reading the tables

**The historical failure and its cause.** The `step` arm reproduces what the reports
found. The longer it lives rule A, the lower its probability of approaching the odour
that will carry the sugar: under rule B its behaviour approached that odour with a median
probability of 0.94, 0.91, 0.63, 0.16 and 0.34 across the exposures, and its lives end
rule B with 90% of their last 100 actions optimal in 9, 10, 9, 5 and 3 of 10. Of the eight
`step` lives whose greedy choice never turned, seven had not approached the new sugar
odour once and one had approached it once; in the 42 lives whose choice turned, one
approach sufficed in all but three, which needed two. The failure is too few
contradicting witnesses; at this operating point a witnessed outcome is taken.

A lag is the first window of 40 executed actions with 36 optimal. A brain that avoids
both odours of the reversal pair is right in three trials of four and can meet such a
window by chance, so for arms that end rule B below 0.9 the lag tables count more lives
than reversed. The last-100 shares, the greedy tables and the behaviour table are the
stricter readings.

**What arousal changes.** Under rule B the `live` arm's behaviour approaches the new sugar
odour with a median probability of 0.94 to 0.98 at every exposure: it executes its greedy
choice with certainty in routine (median executed probability 0.99 over the rule) and
samples when roused. Its first approach comes after 12 to 31 trials in the median life
whatever the exposure, the outcome of that first approach turns its greedy choice in every
life whose choice turned (48 of 50), and its median reversal lag is 15 to 30 trials at
exposures of 300 and more. It lives routine for 90% to 99% of its moments at those
exposures.

**What carries the adaptation.** `memory-only` matches `live`: with the actor's rates at
zero the associative memory and the critic carry acquisition, reversal and return.
`graph-only` ends the rules between 0.52 and 0.80 and its behaviour at the new sugar odour
wanders: the graph's own reward learning does not acquire this task in one stream at these
budgets. `defaults` stays at chance.

**No adaptation without new evidence.** `frozen` and `replay` end rule B at chance for
the reversal pair (0.47 to 0.52 overall) and are right again at once when rule A returns.
`replay` presents its own witnessed records of rule A to its memory 1,200 times over the
two later rules, one per trial, and writes them: equal presentations on old evidence
change nothing about rule B.

**Against starting over.** A newborn brain at every change (`reset`) reverses in 69 to 85
trials, inside its sampling youth, and has the stable pair right at 92% to 96% of the
probes, because it learns that pair again each time; three brains are built. `live`
reverses sooner and keeps the pair.

**Against the table.** At exposures of 300 and more the tabular learner ends the rules at
0.91 to 0.96, the price of exploring a tenth of its actions for life, and reverses with a
median lag of 39 to 92 trials. It needs no brain for four odours and it is the
matched-information reference. `live` ends the rules at 1.00 in the lives that pass and
reverses in 15 to 30 trials: its exploration is raised when reward is missing and absent
otherwise.

**Work.** At exposure 10,000 a `live` life answers 11,200 moments. A routine moment settles
once, 32 sweeps, and an aroused moment takes about 21 sweeps for its answer; its
eligibility and feedback sweeps come to about 1,500 over the life, against 138,000 for
`step`, which learns at every moment. `live` reads its memory about 11,400 times and
writes it about 160 times; `step` reads 22,400 times and writes 11,200. The probes of the
instrument cost about 57,600 sweeps and 2,250 checkpoint files per life, the same for
every brain arm, and are counted apart from the life. No attempt was refused. On this
laptop a routine moment of `live` took 0.50 ms at the median and 0.69 ms at the 90th
percentile, an aroused moment 1.24 and 1.62 ms, and a moment of `step` 1.51 and 2.15 ms;
wall time on one loaded machine is indicative only. Routine saves the eligibility phases,
the parameter updates and the memory writes; it saves no settling work, since every moment
pays one full qualified settle. Sweeps count numerical work; they are neither wall time
nor energy.

### The record forecast, on this freeze's seeds

The second surprise channel, the error of the record the brain held for its chosen
action, at weight one against the founders' zero (`variant-record-surprise-2026-10-06.json.gz`,
`frozen_protocol: false`). On the development seeds this channel left more lives
searching too briefly; here it passed every reading in 40 of 40 lives against 39 of 40,
with median reversal lags of 4.5 to 12 trials. The two samples together give 130 of 136
lives passing with the channel and 133 of 136 without it, and the founders stay as frozen.

| Arousal law | Exposure | Rule A | Rule B | Rule A again | reversal lag | stable pair | aroused |
| --- | --- | --- | --- | --- | --- | --- | --- |
| founders (value forecast) | 300 | 1.00 (1.00) | 1.00 (0.99) | 1.00 (0.99) | 15 (10/10) | 1.00 (1.00) | 0.097 |
| founders (value forecast) | 1,000 | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 24 (10/10) | 1.00 (1.00) | 0.066 |
| founders (value forecast) | 3,000 | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 30 (10/10) | 1.00 (1.00) | 0.036 |
| founders (value forecast) | 10,000 | 1.00 (0.98) | 0.97 (0.67) | 0.98 (0.76) | 16 (9/10) | 1.00 (1.00) | 0.014 |
| record channel weighed one | 300 | 1.00 (1.00) | 1.00 (0.98) | 1.00 (1.00) | 5 (10/10) | 1.00 (1.00) | 0.090 |
| record channel weighed one | 1,000 | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 4 (10/10) | 1.00 (1.00) | 0.057 |
| record channel weighed one | 3,000 | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 12 (10/10) | 1.00 (1.00) | 0.037 |
| record channel weighed one | 10,000 | 1.00 (0.98) | 1.00 (1.00) | 1.00 (1.00) | 6 (10/10) | 1.00 (1.00) | 0.015 |

## Declared variants of the second freeze, 2026-10-05

These runs, on the second freeze's confirmation seeds and source, change one thing each, at exposures of 300, 3,000 and 10,000, and are marked
`frozen_protocol: false`. They were declared before they were run and carry no gate; the
frozen gates are quoted where they help. Cells are the mean (minimum) share of optimal
actions in the last 100 trials of each rule, the median reversal lag with the lives that
have one, the stable pair at the probes, and the median share of aroused moments.
`--report` prints each receipt's full tables. At these three exposures 29 of the 30
confirmation lives of the `live` arm pass every reading of the gates.

### The world

A fifth of the outcomes withheld (`variant-unreliable-2026-10-05.json.gz`, `variant-unreliable-tabular-2026-10-05.json.gz`):

| Arm | Exposure | Rule A | Rule B | Rule A again | reversal lag | stable pair |
| --- | --- | --- | --- | --- | --- | --- |
| `live` | 300 | 1.00 (0.97) | 1.00 (1.00) | 1.00 (1.00) | 27 (10/10) | 0.98 (0.62) |
| `live` | 3,000 | 0.99 (0.92) | 0.96 (0.74) | 1.00 (0.97) | 41 (10/10) | 0.97 (0.50) |
| `live` | 10,000 | 1.00 (1.00) | 1.00 (0.99) | 1.00 (1.00) | 39 (10/10) | 0.99 (0.75) |
| `step` | 300 | 0.81 (0.74) | 0.90 (0.69) | 0.90 (0.79) | 68 (10/10) | 0.95 (0.75) |
| `step` | 3,000 | 0.92 (0.45) | 0.74 (0.47) | 0.92 (0.47) | 227 (5/10) | 0.80 (0.00) |
| `step` | 10,000 | 0.94 (0.57) | 0.65 (0.43) | 0.81 (0.43) | 175 (4/10) | 0.76 (0.00) |
| `tabular`, selected for this world | 300 | 0.90 (0.71) | 0.94 (0.87) | 0.96 (0.95) | 26 (10/10) | 0.93 (0.00) |
| `tabular`, selected for this world | 3,000 | 0.95 (0.90) | 0.95 (0.89) | 0.95 (0.93) | 30 (10/10) | 0.99 (0.92) |
| `tabular`, selected for this world | 10,000 | 0.94 (0.89) | 0.96 (0.91) | 0.95 (0.92) | 74 (10/10) | 1.00 (0.98) |
| `tabular`, protocol settings | 300 | 0.54 (0.48) | 0.56 (0.40) | 0.54 (0.45) | 353 (3/10) | 0.14 (0.00) |
| `tabular`, protocol settings | 3,000 | 0.56 (0.42) | 0.58 (0.47) | 0.58 (0.50) | 238 (1/10) | 0.16 (0.00) |
| `tabular`, protocol settings | 10,000 | 0.56 (0.48) | 0.58 (0.45) | 0.52 (0.42) | 185 (4/10) | 0.18 (0.08) |

Cost payoff with reward noise of 0.03 (`variant-cost-2026-10-05.json.gz`, `variant-cost-tabular-2026-10-05.json.gz`):

| Arm | Exposure | Rule A | Rule B | Rule A again | reversal lag | stable pair |
| --- | --- | --- | --- | --- | --- | --- |
| `live` | 300 | 1.00 (1.00) | 1.00 (0.99) | 0.99 (0.96) | 26 (10/10) | 1.00 (0.96) |
| `live` | 3,000 | 1.00 (1.00) | 1.00 (0.99) | 0.99 (0.93) | 34 (10/10) | 1.00 (1.00) |
| `live` | 10,000 | 1.00 (1.00) | 0.99 (0.91) | 0.98 (0.79) | 34 (10/10) | 0.98 (0.62) |
| `step` | 300 | 0.83 (0.74) | 0.92 (0.69) | 0.97 (0.79) | 55 (10/10) | 0.99 (0.71) |
| `step` | 3,000 | 0.99 (0.97) | 0.98 (0.94) | 1.00 (0.97) | 22 (10/10) | 1.00 (0.95) |
| `step` | 10,000 | 0.99 (0.97) | 0.90 (0.72) | 0.97 (0.85) | 32 (10/10) | 0.97 (0.78) |
| `tabular`, selected for this world | 300 | 0.97 (0.95) | 0.97 (0.94) | 0.96 (0.95) | 0 (10/10) | 1.00 (1.00) |
| `tabular`, selected for this world | 3,000 | 0.97 (0.93) | 0.98 (0.95) | 0.98 (0.96) | 1 (10/10) | 1.00 (1.00) |
| `tabular`, selected for this world | 10,000 | 0.98 (0.95) | 0.98 (0.95) | 0.97 (0.93) | 2 (10/10) | 1.00 (1.00) |
| `tabular`, protocol settings | 300 | 0.95 (0.92) | 0.94 (0.87) | 0.96 (0.95) | 1 (10/10) | 1.00 (1.00) |
| `tabular`, protocol settings | 3,000 | 0.95 (0.90) | 0.95 (0.89) | 0.95 (0.93) | 2 (10/10) | 1.00 (1.00) |
| `tabular`, protocol settings | 10,000 | 0.94 (0.89) | 0.96 (0.91) | 0.95 (0.92) | 11 (10/10) | 1.00 (1.00) |

With a fifth of the outcomes withheld, 28 of 30 `live` lives reverse and all return, and
24 keep the stable pair: a withheld reward at the stable sugar odour takes half of its
record, and the frozen gate on the stable pair is missed. The always-learning arm loses
the reversal with exposure (8, 4 and 1 of 10 lives end rule B at 0.9) and the stable pair
with it. The table is run twice: with the settings the protocol selected in the reliable
world it takes each withheld outcome at face value and stays near chance, and with the
settings the same rule selects in this world (alpha 0.5, epsilon 0.1) it ends the rules
at 0.90 to 0.96.

With the payoff as a cost, every wrong action is punished and carries its own evidence.
29 of 30 `live` lives pass every reading, with a median reversal lag of 26 to 34 trials.
The always-learning arm reverses about as fast here (22 to 55 trials) and ends rule B at
0.9 in 24 of 30 lives. The table with the settings selected for this world (alpha 1.0,
epsilon 0.05) reverses at once, a median lag of 0 to 2 trials: in this world it is the
fastest learner, and `live` ends the rules highest (0.98 to 1.00).

The two runs of the table with other settings used a copy of the protocol with those two
values; their receipts carry it and verify without `--current`.

### The arousal law

| Arousal law | Exposure | Rule A | Rule B | Rule A again | reversal lag | stable pair | aroused |
| --- | --- | --- | --- | --- | --- | --- | --- |
| founders | 300 | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 22 (10/10) | 1.00 (1.00) | 0.097 |
| founders | 3,000 | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 20 (10/10) | 1.00 (1.00) | 0.035 |
| founders | 10,000 | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 22 (10/10) | 0.99 (0.79) | 0.016 |
| no want (`fast` at `slow`) | 300 | 1.00 (1.00) | 0.49 (0.45) | 1.00 (1.00) | none (0/10) | 1.00 (1.00) | 0.067 |
| no want (`fast` at `slow`) | 3,000 | 1.00 (1.00) | 0.52 (0.45) | 1.00 (1.00) | none (0/10) | 1.00 (1.00) | 0.024 |
| no want (`fast` at `slow`) | 10,000 | 1.00 (1.00) | 0.52 (0.42) | 1.00 (1.00) | none (0/10) | 1.00 (1.00) | 0.009 |
| no surprise (`tolerance` 20, `floor` 1) | 300 | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 25 (10/10) | 1.00 (1.00) | 0.105 |
| no surprise (`tolerance` 20, `floor` 1) | 3,000 | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 23 (10/10) | 1.00 (1.00) | 0.039 |
| no surprise (`tolerance` 20, `floor` 1) | 10,000 | 1.00 (1.00) | 0.99 (0.94) | 1.00 (1.00) | 23 (10/10) | 1.00 (1.00) | 0.016 |
| no heat (`heat` 0) | 300 | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 22 (10/10) | 1.00 (1.00) | 0.100 |
| no heat (`heat` 0) | 3,000 | 1.00 (1.00) | 0.98 (0.80) | 0.98 (0.76) | 22 (10/10) | 1.00 (1.00) | 0.035 |
| no heat (`heat` 0) | 10,000 | 1.00 (1.00) | 0.95 (0.77) | 0.95 (0.74) | 21 (9/10) | 1.00 (0.96) | 0.018 |

Without the want no life reverses: the brain stays in routine through rule B, as the
frozen brain does. Without surprise all 30 lives pass every reading, so in this chamber
surprise adds nothing that can be measured. Without the heat 26 of 30 lives pass, against
29 of 30 with the founders; the four misses are at exposures of 3,000 and 10,000.

### The operating point and the size of the brain

| Operating point | Exposure | Rule A | Rule B | Rule A again | reversal lag | stable pair | aroused |
| --- | --- | --- | --- | --- | --- | --- | --- |
| declared | 300 | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 22 (10/10) | 1.00 (1.00) | 0.097 |
| declared | 3,000 | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 20 (10/10) | 1.00 (1.00) | 0.035 |
| declared | 10,000 | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 22 (10/10) | 0.99 (0.79) | 0.016 |
| trace amplitude 3.0 | 300 | 0.74 (0.46) | 0.80 (0.52) | 0.81 (0.41) | 123 (6/10) | 0.59 (0.00) | 0.112 |
| trace amplitude 3.0 | 3,000 | 0.87 (0.49) | 0.82 (0.47) | 0.81 (0.48) | 80 (7/10) | 0.61 (0.00) | 0.054 |
| trace amplitude 3.0 | 10,000 | 0.84 (0.44) | 0.78 (0.43) | 0.67 (0.44) | 38 (6/10) | 0.56 (0.00) | 0.031 |
| consolidation 0.05 | 300 | 1.00 (1.00) | 0.97 (0.71) | 0.97 (0.69) | 27 (9/10) | 1.00 (1.00) | 0.099 |
| consolidation 0.05 | 3,000 | 1.00 (1.00) | 0.99 (0.93) | 0.97 (0.73) | 29 (10/10) | 1.00 (0.92) | 0.036 |
| consolidation 0.05 | 10,000 | 1.00 (0.99) | 0.96 (0.75) | 0.96 (0.77) | 53 (10/10) | 1.00 (0.92) | 0.017 |
| actor rates 1.0 and 0.05 | 300 | 0.78 (0.48) | 0.71 (0.40) | 0.72 (0.41) | 25 (5/10) | 0.47 (0.00) | 0.135 |
| actor rates 1.0 and 0.05 | 3,000 | 0.71 (0.43) | 0.57 (0.40) | 0.60 (0.42) | 87 (2/10) | 0.28 (0.00) | 0.064 |
| actor rates 1.0 and 0.05 | 10,000 | 0.67 (0.46) | 0.62 (0.49) | 0.60 (0.45) | 17 (2/10) | 0.28 (0.00) | 0.018 |
| all three released | 300 | 0.49 (0.41) | 0.49 (0.41) | 0.48 (0.41) | none (0/10) | 0.00 (0.00) | 0.067 |
| all three released | 3,000 | 0.57 (0.43) | 0.60 (0.46) | 0.52 (0.42) | 268 (2/10) | 0.12 (0.00) | 0.024 |
| all three released | 10,000 | 0.54 (0.41) | 0.55 (0.43) | 0.50 (0.44) | 80 (1/10) | 0.08 (0.00) | 0.009 |
| one module of 64 | 300 | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 23 (10/10) | 1.00 (1.00) | 0.098 |
| one module of 64 | 3,000 | 1.00 (1.00) | 1.00 (1.00) | 0.96 (0.64) | 20 (10/10) | 1.00 (1.00) | 0.036 |
| one module of 64 | 10,000 | 1.00 (1.00) | 0.98 (0.77) | 0.95 (0.74) | 26 (9/10) | 1.00 (1.00) | 0.016 |
| two modules of 32 | 300 | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 23 (10/10) | 1.00 (1.00) | 0.100 |
| two modules of 32 | 3,000 | 1.00 (1.00) | 1.00 (0.98) | 1.00 (1.00) | 21 (10/10) | 1.00 (0.96) | 0.035 |
| two modules of 32 | 10,000 | 1.00 (1.00) | 0.97 (0.77) | 0.97 (0.74) | 19 (9/10) | 1.00 (1.00) | 0.014 |

The working trace and the actor rate are required: with either at its released value
most lives fail, and with all three released the life stays at chance, arousal included.
The released consolidation works less reliably (25 of 30 lives pass every reading). The
result does not depend on the 32-neuron module: one module of 64 neurons passes every
reading in 27 of 30 lives and two modules of 32 in 29 of 30.

## The earlier freezes

The chamber was first frozen with confirmation seeds 100 to 109 (protocol SHA-256
`381f23ca76287e0b83eeea99d0d5f5a3ae1adba2fce2be2eed0c814706ab1f8f`) and run once on
2026-10-05. Every gate passed: all 40 gated lives of the `live` arm acquired, reversed,
returned, kept the stable pair and returned to routine, with median reversal lags of
22.5, 31.5, 25.5 and 24 trials. Its receipt is
`results/first-freeze-confirmation-2026-10-05.json.gz`; it records the protocol hash and
the library version and predates the source manifest.

A review of the code after that run found that the actor's eligibility traces did not
fade while a brain lived in routine, so the first outcome learned after waking credited
actions sampled before the calm. The library was corrected (`ActorCritic.fade`), the
confirmation seeds of the first freeze were declared spent, and the protocol was frozen
again with fresh seeds, the same world, operating point, arousal founders and gates. The
second freeze also selects the tabular learner's two settings on the development seeds
(the first used alpha 0.2 and epsilon 0.1 unselected), gives a reset arm's newborn brain
a seed that meets no other generator of its life, adds the coverage and witness readings,
and binds its receipts to the source.

The second freeze (protocol SHA-256
`0a5f2b4e6c2e2767aebc57588caf3e338a0348f5ec14ab099f637413f09652c4`, seeds 300 to 309)
ran once on 2026-10-05 at commit
[`9b66228`](https://github.com/muellerberndt/cadence/commit/9b66228) and passed every
gate: of its 40 gated lives 40 acquired, 39 reversed, 38 returned, 39 kept the stable pair
and 40 returned to routine, with median reversal lags of 22.5, 29, 20.5 and 22 trials.
Three lives missed a reading: one at exposure 1,000 never approached the moved sugar and
ended both later rules avoiding the reversal pair; one at exposure 1,000 ended the last
rule at 0.87; one at exposure 10,000 had the stable pair right at 79% of the probes of a
slow return. Its receipt is `results/confirmation-2026-10-05.json.gz`, with the variant
receipts of that day beside it; `--report` prints their tables. An audit on 2026-10-06
then hardened the law's numerics and the receipts without changing the law, and its arm
named `replay` was found to pay current actions by the old rule rather than to replay
witnessed records; its work counters covered successful solver calls only. The third
freeze keeps the law, the operating point and the gates of the second and adds the
behaviour probabilities, the retained-record replay and the complete work ledger that
issue 88 required. The declared variants of the second freeze stand: the law and the
operating point are unchanged, and the readings they report were not changed by the
third freeze.

## How the operating point and the founders were selected

Development used seeds 0 to 23, and seeds 400 to 447 at the longest exposure; the
confirmation seeds were first run on the frozen protocol.

- **The released defaults lock one stream.** With the composed actor rate of 1.0, selected
  on batches of streams, the policy of one stream saturates on one action for every odour
  within about a hundred trials. A tenth of the rate leaves the choice to the evidence.
- **The default working trace outweighs the present input of a continuing life.** With
  amplitude 3.0 into the scale-12 prefrontal projection, 1% of the variance of the
  association state follows the present odour and 1% the previous one; the state follows
  its own history. At amplitude 0.3, 76% follows the present odour and 16% the previous
  one. Amplitude 0 and 0.3 gave the same nursery results.
- **Lasting memory took too little of a witnessed outcome.** At the default consolidation
  of 0.05 a unit outcome moves the lasting record by a tenth while the transient copy
  fades by a tenth with every other record. After 10,000 trials of rule A a life sampled
  the new sugar odour, was paid, and returned to avoiding it before the record had turned:
  a failure to revise after sufficient witnesses. At 0.25 the lasting record takes half of
  a unit outcome. Consolidation 0.5 and 1.0 gave the same results in the reliable world;
  with a fifth of the outcomes withheld, 0.5 lost the stable pair on single withheld
  rewards and 0.25 kept it in all but one development life.
- **Whose outcomes enter the mood.** Letting every outcome enter it made the cost of
  exploring look like a shortfall and kept the brain awake: with the payoff as a cost
  (nothing for the right action, -1 for the wrong one, `--payoff cost`) the median
  reversal lag was 73 to 89 trials and 10% to 16% of the moments after a change were
  aroused. With only the outcomes of the brain's own greedy choices it was 17 to 25
  trials and 3% to 4%. In the sugar payoff the two rules did not differ.
- **The unit of reward.** A running RMS reward shrinks through a long calm at little
  reward and then makes small fluctuations look large. The unit is the spread of the
  outcomes the brain has learned from, which routine outcomes leave alone; the law is
  then unchanged by the scale and the zero of reward.
- **Arousal founders.** Removing the want left lives stuck after the reversal. With the
  hand-set heat of 1, 5 of 84 development lives at exposures of 300 and more ended a rule
  below the gate, having stopped searching before the moved sugar was found; with heat 2,
  and separately with a long-run rate of 0.002, none of 96 did before the eligibility
  correction. Heat 2 was kept as the founder: it spent the smaller share of moments
  aroused. Heat 3, a threshold of 0.3 and a faster recent rate were no better.
- **After the eligibility correction** the founders were left as they were. On the
  development seeds 94 of 96 lives at exposures of 300 and more then passed every reading,
  both misses at exposure 10,000. On the 48 diagnostic seeds at exposure 10,000, 45 lives
  ended every rule at 0.9 with the correction and 46 without it: the correction did not
  change the rate of misses.
- **The action record as a forecast.** The audit of the second freeze asked for a learned
  forecast tied to the chosen action. The record the associative memory holds for the
  chosen action in the situation met is one, and a second surprise channel measures its
  error against its own usual size. On the development seeds at the four gated exposures
  the record channel alone woke the brain sooner (median reversal lag 18 trials against
  23) and left more lives searching too briefly, caught avoiding both odours of the
  reversal pair after one or two punishments: 89 of 96 lives passed every reading with the
  record channel alone and 90 of 96 with both channels, against 94 of 96 with the value
  forecast alone. The founders weigh the record channel at zero, so the law of the second
  freeze is unchanged; the channel is a gene (`record_surprise`) left to selection. On the
  confirmation seeds the same comparison is a declared variant below.
- **The lag gate.** The gate on the reversal lag was first a ratio between the longest and
  the shortest exposure. A ratio of two medians of about 20 trials moves with single
  lives, and before the first confirmation it was replaced by the bound of 150 trials at
  each exposure.
- **The tabular learner.** Alpha in 0.1, 0.2, 0.5 and 1.0 and epsilon in 0.02, 0.05, 0.1,
  0.15, 0.2 and 0.3 were tried on the development seeds at exposures of 300 and more. The
  largest share of lives ending every rule at or above 0.9 was 94%, at epsilon 0.1 with
  alpha 0.5 or 1.0; alpha 1.0 had the lower median reversal lag, 38 trials against 52.

## Income tracking repair — 2026-10-08

Issue [#158](https://github.com/muellerberndt/cadence/issues/158) exposes a different
failure of the historical own-only mood law: a brain exploring with several motor
slots rarely chooses its greedy action in every slot, so its income averages can
remain stuck at an earlier stage. The repair admits every actual reward to recent
and long-run income, while surprise and both usual forecast errors remain own-only.
The settlement, learning equations, action credit and genes are unchanged.

The scalar regression in `tests/test_arousal_income.py` starts both laws with 200
alternating 0/1 outcomes, 600 at 0.5, 60 at 0.25, then 1,500 sampled outcomes at
0.25. It supplies no surprise or reset. The previous law ends with want 0.9367
and is aroused for all of the last 300 outcomes; the repair ends with want 0.0106
and is calm for those 300. Its income approaches 0.25 while both laws retain
860 own outcomes for error calibration. A positive unmet need still persists.
This is an income-law regression, not a learned-skill result.

The historical costly-exploration finding above remains relevant. To check its
preserved behavioral gates, five paired development founders (0–4) were run on
the cost payoff with exposure 300 and 600 trials in each later rule. The control
is source `2566ccf`; the repaired runtime is `9c2ac1f` (also integrated as
`b041dc7`). The same observations, world, genes, memory, reward rules and work
accounting apply. Each side contains five continuing lives and 7,505 issued
actions. Seeds 0–1 ran first; the declared extension then added 2–4 without
tuning. These are reused development seeds, not fresh confirmation, and this
single exposure does not repeat the original full confirmation census.

| Reading across the five lives | Own-only income control | All-outcome income repair |
| --- | ---: | ---: |
| Lives meeting every measured acquisition/reversal/return/stable/calm gate | 5/5 | 5/5 |
| Final executed accuracy and stable-pair probes, every phase | 1.00 | 1.00 |
| Late aroused share, every phase | 0.00 | 0.00 |
| Median acquisition / reversal / return lag | 83 / 36 / 35 | 83 / 68 / 77 |
| Aroused actions / all issued actions | 863 / 7,505 (11.50%) | 1,495 / 7,505 (19.92%) |
| Routine / aroused settling sweeps | 212,256 / 18,048 | 192,160 / 33,440 |
| Learning / probe sweeps | 9,606 / 39,680 | 18,108 / 39,808 |
| Memory writes / reads | 873 / 8,378 | 1,506 / 9,011 |
| Refused sweeps | 0 | 0 |

The repair costs more exploration and slower recovery here. Seed 2's return lag
increases from 53 to 225 trials. The unchanged gates require final accuracy at
least 0.9, stable-pair correctness at least 0.95 and late arousal at most 0.2 in
at least 90% of lives, plus median **reversal** lag at most 150; they do not bound
each individual return lag. No gate was relaxed. This preserves those bounded
capabilities while repairing the stale-income defect; it does not establish a
speed, efficiency or robot-fighting improvement.

The four small source-bound receipts retain both halves of each comparison:
[control 0–1](results/development-income-control-initial-2026-10-08.json.gz),
[repair 0–1](results/development-income-candidate-initial-2026-10-08.json.gz),
[control 2–4](results/development-income-control-additional-2026-10-08.json.gz),
[repair 2–4](results/development-income-candidate-additional-2026-10-08.json.gz).
All passed the existing digest/source/arithmetic verifier. An initial control
attempt was rejected because source files changed while it ran; its results
were not admitted, and the control was rerun in an unchanged checkout. Reproduce
each pair at its stated source with:

```bash
python benchmarks/reversal/odour_nursery.py --arms live --seeds 0 1 --exposures 300 --payoff cost --workers 1 --out income-initial.json.gz
python benchmarks/reversal/odour_nursery.py --arms live --seeds 2 3 4 --exposures 300 --payoff cost --workers 1 --out income-additional.json.gz
```

## Temperature-consistent credit and slot exploration — 2026-10-08

The combined repair at `6fae103` adds two separate changes to the income repair:
actor eligibility now uses the policy that actually sampled the action, and
extra arousal heat applies to one uniformly chosen motor slot while other slots
sample the base policy. No new genes or configured default values are introduced.
The local actor nudge uses
the actual temperature and a bounded mask gain, so every output contributes to
the same scaled joint policy score. Finite phases retain their approximation
error. The independent derivative and action-independent-baseline checks are
in `tests/test_behavior_credit.py`; sampling, refusal and saved-continuation
checks are in `tests/test_arousal_slots.py`.

The same five development founders (0–4), cost payoff, exposure 300 and 600-trial
later rules were checked against the original gates, without tuning. The
[combined receipt](results/development-temperature-combined-2026-10-08.json.gz)
passes its digest/source/arithmetic verifier. This is bounded preservation on
reused development founders, not a new confirmation census or a robot-skill claim.

| Reading across the five lives | Income repair alone (`9c2ac1f`) | Combined repair (`6fae103`) |
| --- | ---: | ---: |
| Lives meeting all measured acquisition/reversal/return/stable/calm gates | 5/5 | 5/5 |
| Final executed accuracy and stable-pair probes, every phase | 1.00 | 1.00 |
| Late aroused share, every phase | 0.00 | 0.00 |
| Median acquisition / reversal / return lag | 83 / 68 / 77 | 83 / 68 / 129 |
| Longest individual return lag | 225 | 239 |
| Aroused actions / all issued actions | 1,495 / 7,505 (19.92%) | 1,659 / 7,505 (22.11%) |
| Routine / aroused settling sweeps | 192,160 / 33,440 | 186,912 / 33,792 |
| Learning / probe sweeps | 18,108 / 39,808 | 18,042 / 39,808 |
| Memory writes / reads | 1,506 / 9,011 | 1,670 / 9,175 |
| Refused sweeps | 0 | 0 |

The combined repair retains the declared capabilities but returns more slowly
in this cost task. Original gates remain final accuracy at least 0.9,
stable-pair correctness at least 0.95, late arousal at most 0.2 in at least 90%
of lives, and median reversal lag at most 150. There is no individual return-lag
gate. The slower returns and all earlier comparisons above remain recorded;
these data do not support a recovery-speed improvement.

The sampler change leaves a one-slot life unchanged. A separate
[credits-only control](results/development-temperature-credit-only-2026-10-08.json.gz)
at `7e308f6`, seed 0, matches the combined seed 0 exactly in every recorded phase
and every work field except latency. Both acquire/reverse/return in 89/68/82
trials. This isolates the sampler from the credit correction; it is not a
comparison between five independent new tasks. The control receipt also passes
the verifier. The source manifests in both receipts identify every runtime file.

The multi-slot sampling test separately holds bounded motor preferences fixed
and reads the actual live-selected temperatures. With five three-state slots,
activation margin 1, base temperature 0.5 and heated temperature 1.5, the exact
probability of the preferred complete command is 0.02924 when all slots are
heated and 0.18926 when one slot is heated (0.30188 without extra heat). This is
a motor-boundary control, not acquired behavior. For `S` slots, each slot's
marginal distribution differs from its base policy by only `1/S` of the change
caused by heating that slot every moment; all alternatives remain available.

Reproduction at each stated source uses the earlier cost-payoff command with
`--seeds 0 1 2 3 4` for the combined receipt and `--seeds 0` for the credits-only
control. Both use one worker and the original protocol and operating point.

## The readout's intrinsic plasticity at rate 0.02 (development, 2026-10-11)

The learner gene `homeostasis_rate` (founder 0) was added for the
[recall chamber](../recall/README.md). Its effect on this nursery was read on the six
development founders 0 to 5, the `live` arm, the protocol's sugar payoff and all five
exposures, once at the founder and once with the gene at 0.02, on the same source:
[founder receipt](results/development-homeostasis-founder-2026-10-11.json.gz) and
[gene receipt](results/development-homeostasis-0.02-2026-10-11.json.gz), both verified.
This is a development reading on reused founders with no gate.

| Reading over six lives, by exposure | 100 | 300 | 1,000 | 3,000 | 10,000 |
| --- | --- | --- | --- | --- | --- |
| Rule B, optimal share of the last 100 actions, founder | 1.00 (1.00) | 1.00 (1.00) | 0.99 (0.94) | 1.00 (1.00) | 1.00 (1.00) |
| Rule B, optimal share of the last 100 actions, gene 0.02 | 0.86 (0.71) | 1.00 (1.00) | 0.99 (0.95) | 1.00 (1.00) | 0.99 (0.95) |
| Reversal lag, median (lives that reversed), founder | 54 (6/6) | 20 (6/6) | 31 (6/6) | 74 (6/6) | 65 (6/6) |
| Reversal lag, median (lives that reversed), gene 0.02 | 47 (5/6) | 28 (6/6) | 32 (6/6) | 29 (6/6) | 26 (6/6) |
| Return lag, median (lives that returned), founder | 34 (6/6) | 22 (6/6) | 21 (6/6) | 27 (6/6) | 32 (6/6) |
| Return lag, median (lives that returned), gene 0.02 | 80 (5/6) | 22 (6/6) | 21 (6/6) | 14 (6/6) | 21 (6/6) |
| Lives whose greedy probes all turned right after the reversal, founder | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 |
| Lives whose greedy probes all turned right after the reversal, gene 0.02 | 3/6 | 6/6 | 6/6 | 6/6 | 6/6 |
| Approach to the new sugar odour under rule B, median (minimum), founder | 0.95 (0.81) | 0.96 (0.93) | 0.96 (0.95) | 0.95 (0.87) | 0.96 (0.93) |
| Approach to the new sugar odour under rule B, median (minimum), gene 0.02 | 0.42 (0.00) | 0.95 (0.94) | 0.96 (0.92) | 0.96 (0.92) | 0.94 (0.91) |

The stable pair reads 1.00 at every probe of both arms. From an exposure of 300 trials the
gene keeps every measured capability, and after the long exposures it reverses faster
(median 29 and 26 trials at 3,000 and 10,000 against 74 and 65). After the shortest
exposure it impairs reversal: one life of six neither reverses nor returns, the greedy probes
turn right in three lives of six, and the median approach to the new sugar odour under rule B
falls from 0.95 to 0.42. The gene is not neutral in this nursery; it stays off by default.
Reproduction:

```sh
PYTHONPATH=src python benchmarks/reversal/odour_nursery.py --arms live --seeds 0 1 2 3 4 5 \
    --workers 6 --out <founder receipt>
PYTHONPATH=src python benchmarks/reversal/odour_nursery.py --arms live --seeds 0 1 2 3 4 5 \
    --workers 6 --point '{"learning": {"homeostasis_rate": 0.02}}' --out <gene receipt>
```

## What this does and does not establish

The `live` arm supplies bounded acquisition, reversal, return and stable-pair evidence
for issue 88. The choice that
must change is sampled again because a lasting shortfall of reward rouses the brain and
widens its sampling, and one witnessed outcome turns the choice because the lasting
record takes half of it. The always-learning arm shows the historical failure on the same
sequences, and its cause.

The readings issue 88 still required after the second freeze are in this one: the
probabilities of the behaviour that acted, read from the living brain at every trial; a
replay control that presents the brain's own witnessed records again, with its
presentations counted; and the complete work of a life, probes, memory operations,
construction, refused attempts and wall time included.

The repair is incomplete. Three of the 40 confirmation lives missed a reading, two at
exposure 1,000 and one at 10,000; on the development and diagnostic seeds 5 of 72 lives
at exposure 10,000 and none of 72 at exposures of 300 to 3,000 did. The founders' margin
is thin: in the confirmation life that never reversed, the arousal stayed under its
threshold for about 60 trials while the old sugar odour punished nine approaches; the
outcome that then woke the brain was recorded, the brain came to avoid that odour, the
shortfall that remained was within the threshold, and it never searched for the moved
sugar. The want
fades as the brain grows used to the poorer life, and a brain that avoids both odours of
the reversal pair is paid what it forecasts and stays calm. The dependence on exposure
that the issue describes is reduced and is still present.

The associative memory carries the adaptation. The odours are one-hot and the memory is a
direct record from sensory keys to action values, so this chamber does not test the
reciprocal graph's learned relations, generalization to unseen inputs, delayed outcomes
or short-term recall; those remain with issues
[110](https://github.com/muellerberndt/cadence/issues/110),
[111](https://github.com/muellerberndt/cadence/issues/111),
[84](https://github.com/muellerberndt/cadence/issues/84) and
[121](https://github.com/muellerberndt/cadence/issues/121). The declared operating point
is required: with arousal on the released composition the life stays at chance.

In this chamber the want does the rousing. The founders' forecast is the critic's value
of the situation, and the next odour is random, so the usual error of a correct forecast
is about half a reward; a contradicted approach exceeds twice that by little and adds
little surprise. With the want removed no life reverses, and with surprise removed every
life passes. A forecast specific to the chosen action, the record the memory holds for
it, is measured and available as a gene; at its founder weight of zero it does nothing,
because with it the brain woke sooner and searched less.

The arousal law responds to change. In a world that does not change, noise in the reward
rate still rouses the founder genes for a small share of moments (the table reports it),
and a bout of needless exploration can lower the last 100 trials of a rule below the
gate. A brain whose life has always paid poorly, and whose youth has ended, is not roused.
The arousal level is a scalar computed from the brain's own temporal-difference error and
reward, as its dopamine is; it is a hand-set law whose constants are genes, and a
settling arousal patch inside the graph is not attempted here.

Issue 122 remains open. Routine is not cheaper in settling work, and the wall time of a
moment here is a single laptop's. Reuse of a settled state across moments,
dependency-local repair and their invalidation tests are not attempted. Its broader gates
also require changed goals and delayed outcomes, familiar nonidentical inputs, memory
interventions beyond the arms here, and independent preservation evidence beyond this
bounded chamber.

Sweeps count numerical work. They are not wall time or energy.
