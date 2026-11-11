# Night-replay chamber

This bounded instrument asks whether a continuing `Brain.compose` brain gains from
re-experiencing its own day through `step`, the offline replay that
[issue 139](https://github.com/muellerberndt/cadence/issues/139) reported as
collapsing a greedy policy to the resting action, and it supplies the
equal-experience comparison of awake use against replay that
[issue 112](https://github.com/muellerberndt/cadence/issues/112) asks for. It tests
one composed brain on one synthetic world; it does not choose defaults, establish a
consolidation mechanism or complete either contract. Read the
[reward guide](../../docs/reward.md#replaying-a-life-through-step) and the
[world-model guide](../../docs/start.md#one-continuing-equilibrium-brain) before interpreting results.

## What runs

The world is one decision per interval. A cue in {-1, +1} sits in the observation
beside two lagged outcomes, their momentum and magnitude, the time of day and a
constant. The interval's outcome is signed, in small units. The brain can stay, move
left or move right; moving left earns the outcome less a cost of 2, moving right the
negated outcome less the cost, staying nothing. In the `signal` world the outcome is
four times the cue plus noise, so the cue decides which move pays; in the `noise`
world the outcome is noise and the cue a coin, so staying is the best policy. This is
the contract of the reported paper-trading loop: hold, buy and sell; an outcome in
basis points; a fixed cost per trade. Nothing here is a trading claim.

For each seed and setting, `night_replay.py` runs:

1. **The day.** `days` decisions on one stream, `step(observation, reward=...)` with
   the reward of the preceding sampled action.
2. **The night.** A saved copy replays `rounds` rounds of the day: the original day,
   its mirror (directional inputs and outcomes negated) and a half-interval shift.
   Each pass starts by paying the reward owed to the pending action with
   `done=True`, so passes are separate episodes. Nothing but `step` is called; the
   copy cannot tell the night from a day.
3. **The awake control.** Another saved copy continues for exactly as many decisions
   on fresh days of the same world.
4. **The readings**, each on a frozen copy (save, load, `reset`), before the night,
   after it and after the awake control: the greedy choice on each of the day's
   observations in order, with the trace carried as in life; the per-observation
   policies and their *dependence* on the observation, the mean total variation
   between each policy and the mean policy (0 when the choice ignores the
   observation, near 0.5 here when it follows the cue); the *cue agreement*, the share
   of greedy choices that make the paying move in the signal world; the graph-only
   choice (`predict`, without trace or memory); the memory's mean recall drive per
   action; and the shares of outcomes the dopamine cap clipped (`report["capped"]`)
   and of saturated motor outputs during the day.

The brain is `Brain.compose(8, 3, observers=(8,))` with the reported configuration
(`consolidation=0.05`, `gamma=0.9`, `lam=0.8`). The settings change only what they
name: `trace-1-0.8` sets `working_memory_amplitude=1.0, working_memory_decay=0.8`,
`trace-off` sets the amplitude to zero, `actor-0.1` and `actor-0.03` set the actor's
`eta` with `eta_bias` a tenth of it, `no-memory` passes `episodic=False`,
`frozen-actor` zeroes the actor and critic rates, `no-cost` removes the cost of a
move and `reward/5` divides every reward by five. The `centered` rows of the tables set
`dopamine_center=0.9`, the centred dopamine that the library removed after 0.84.3; the
arm runs on releases up to 0.84.3, and its readings stay as recorded.

```bash
python benchmarks/replay/night_replay.py --seeds 5 --days 40 --rounds 3 --workers 4 --out night.json
python benchmarks/replay/night_replay.py --world signal --settings defaults,trace-1-0.8+actor-0.1
```

The protocol test in `tests/test_replay_chamber.py` checks that the readings leave the
live brain byte-identical, that the night and the awake control consume the same
number of decisions, and that settings change only the named parameters.

## Measured: a 40-decision day, three rounds, five seeds

Cadence 0.73.1 source with the `capped` reading; cadence-net 0.72.1 gives the same
numbers, since the composed settlement and learning are unchanged for three actions,
except the `capped` column of `centered`, where the chamber's fallback on releases
without the reading counts the raw error against the cap rather than the centred
signal that is clipped (0.32 and 0.26 there).
Cells are means over five seeds; "night at 1.00" counts seeds whose greedy choice
agreed with the cue rule on every observation after the night.

### Signal world: the cue decides the paying move

| Setting | Dependence before / night / awake | Cue agreement before / night / awake | Night at 1.00 (seeds) | Capped (day) | Saturation (day) |
| --- | --- | --- | --- | --- | --- |
| `defaults` | 0.24 / 0.06 / 0.31 | 0.60 / 0.39 / 0.62 | 0 of 5 | 0.43 | 0.76 |
| `trace-1-0.8` | 0.32 / 0.30 / 0.15 | 0.77 / 0.75 / 0.65 | 2 of 5 | 0.63 | 0.71 |
| `trace-off` | 0.38 / 0.38 / 0.48 | 0.79 / 0.80 / 0.99 | 4 of 5 | 0.55 | 0.67 |
| `actor-0.1` | 0.43 / 0.46 / 0.48 | 0.78 / 0.98 / 0.99 | 4 of 5 | 0.54 | 0.64 |
| `trace-1-0.8+actor-0.1` | 0.44 / 0.48 / 0.48 | 0.90 / 1.00 / 1.00 | 5 of 5 | 0.66 | 0.68 |
| `trace-1-0.8+actor-0.03` | 0.46 / 0.48 / 0.48 | 0.91 / 1.00 / 0.99 | 5 of 5 | 0.63 | 0.64 |
| `trace-off+actor-0.1` | 0.47 / 0.48 / 0.47 | 0.96 / 1.00 / 0.99 | 5 of 5 | 0.69 | 0.68 |
| `no-memory+trace-1-0.8+actor-0.1` | 0.07 / 0.11 / 0.05 | 0.31 / 0.12 / 0.25 | 0 of 5 | 0.70 | 0.65 |
| `no-memory` | 0.01 / 0.00 / 0.01 | 0.38 / 0.44 / 0.29 | 0 of 5 | 0.65 | 0.81 |
| `frozen-actor` | 0.45 / 0.48 / 0.48 | 0.86 / 0.96 / 0.89 | 3 of 5 | 0.56 | 0.62 |
| `no-cost` | 0.02 / 0.06 / 0.07 | 0.28 / 0.33 / 0.28 | 0 of 5 | 0.46 | 0.86 |
| `reward/5` | 0.02 / 0.00 / 0.00 | 0.17 / 0.27 / 0.17 | 0 of 5 | 0.12 | 0.73 |
| `centered` | 0.15 / 0.13 / 0.31 | 0.39 / 0.52 / 0.62 | 1 of 5 | 0.19 | 0.74 |

### Noise world: nothing pays, staying is correct

| Setting | Dependence before / night / awake | Greedy after the night (stay, left, right; summed over seeds) | Capped (day) | Saturation (day) |
| --- | --- | --- | --- | --- |
| `defaults` | 0.02 / 0.01 / 0.00 | 200, 0, 0 | 0.19 | 0.74 |
| `trace-1-0.8` | 0.05 / 0.01 / 0.01 | 161, 0, 39 | 0.20 | 0.70 |
| `trace-off` | 0.02 / 0.00 / 0.00 | 200, 0, 0 | 0.23 | 0.65 |
| `actor-0.1` | 0.04 / 0.01 / 0.01 | 200, 0, 0 | 0.24 | 0.65 |
| `trace-1-0.8+actor-0.1` | 0.11 / 0.01 / 0.02 | 200, 0, 0 | 0.25 | 0.64 |
| `trace-1-0.8+actor-0.03` | 0.12 / 0.04 / 0.03 | 199, 1, 0 | 0.36 | 0.70 |
| `trace-off+actor-0.1` | 0.21 / 0.00 / 0.00 | 200, 0, 0 | 0.41 | 0.64 |
| `no-memory+trace-1-0.8+actor-0.1` | 0.07 / 0.01 / 0.04 | 200, 0, 0 | 0.37 | 0.58 |
| `no-memory` | 0.01 / 0.00 / 0.00 | 160, 40, 0 | 0.24 | 0.73 |
| `frozen-actor` | 0.17 / 0.14 / 0.12 | 156, 15, 29 | 0.38 | 0.65 |
| `no-cost` | 0.01 / 0.03 / 0.02 | 1, 81, 118 | 0.40 | 0.79 |
| `reward/5` | 0.01 / 0.00 / 0.01 | 120, 41, 39 | 0.07 | 0.77 |
| `centered` | 0.02 / 0.03 / 0.02 | 93, 56, 51 | 0.18 | 0.76 |

## What the readings say

- **At the defaults the choice ignores the observation before any night.** Dependence
  0.02 without a signal, and in the signal world the night lowered the cue agreement
  from 0.60 to 0.39 while the same number of fresh decisions kept it at 0.62. The
  collapse the issue reports is a replay of a day whose updates were already
  destroying what the day had learned.
- **The cost selects the held action; it does not cause the lock.** Without a cost
  (`no-cost`) the noise-world night still ends on one action per seed, a different
  one each time (1, 81, 118), and the signal-world agreement stays at 0.28 to 0.33.
- **Outcomes beyond the cap teach their sign alone.** `capped` is 0.4 to 0.7 of the
  day's outcomes at the default `dopamine_cap=1.0`: a gain of one unit and a loss of
  six move every synapse by the same amount. Dividing the rewards by five
  (`reward/5`) lowers `capped` to 0.1 but at the actor rate of 1.0 the policy is
  constant all the same (dependence 0.02): with a cost of 0.4 against gains of 0.8
  every move loses on average until the cue is learned, and the actor reaches the
  resting action first.
- **The actor rate of 1.0 on one stream walks the policy to a held action; 0.1 lets
  the night help.** At `actor-0.1` the night raised the agreement from 0.78 to 0.98
  (4 of 5 seeds at 1.00), as the awake control did; with the trace at 1.0/0.8 or off
  as well, from 0.90 or 0.96 to 1.00 in every seed. In the noise world the same
  settings end on `stay` in every seed, the correct answer there. The composed rate
  was selected on batches of 32 streams, where the update is an average over the
  batch; one stream gets the same rate on one sample.
- **The associative memory is the fast learner of this task.** With the actor and
  critic frozen the memory alone reached 0.86 agreement in the day and 0.96 after
  the night: it stores the outcome of the chosen move keyed by the observation, and
  the cue is in the key. Without the memory the actor at 0.1 reached 0.31 and the
  night lowered it to 0.12. In the noise world the same memory leans the choice to
  `stay` (156 of 200 after the night under a frozen actor), since a move that lost
  is recalled as a loss.
- **The working trace.** On this task the trace at 1.0/0.8 raised the day's
  agreement from 0.60 to 0.77 at the default actor rate and from 0.78 to 0.90 at
  0.1; on the continuing bandit below it is the deciding setting.
- **A night is worth about as much as the same number of fresh decisions where the
  day's updates are sound, and less than nothing where they are not.** At
  `trace-1-0.8+actor-0.1` the night and the awake control both reached 1.00; at the
  defaults the night reached 0.39 against 0.62 awake. The night beat fresh days most
  under the frozen actor (0.96 against 0.89): the memory's consolidation rule rewards
  repetition of the same keys, which fresh days do not supply.

## A longer day and a longer night

Cadence-net 0.72.1, five seeds, the settings that matter; the day's agreement and
the night's, with the awake control.

| Setting | 144 decisions, 3 rounds: before / night / awake | 40 decisions, 10 rounds: before / night / awake |
| --- | --- | --- |
| `defaults` (signal, cue agreement) | 0.12 / 0.20 / 0.12 | 0.60 / 0.32 / 0.54 |
| `trace-1-0.8` (signal, cue agreement) | 0.56 / 0.41 / 0.60 | 0.77 / 0.64 / 0.63 |
| `trace-off` (signal, cue agreement) | 0.78 / 0.59 / 0.80 | 0.79 / 0.80 / 1.00 |
| `trace-1-0.8+actor-0.1` (signal, cue agreement) | 1.00 / 0.88 / 1.00 | 0.90 / 1.00 / 1.00 |
| `trace-1-0.8+actor-0.03` (signal, cue agreement) | 0.93 / 1.00 / 1.00 | 0.91 / 1.00 / 1.00 |
| `trace-off+actor-0.1` (signal, cue agreement) | 1.00 / 1.00 / 1.00 | 0.96 / 0.97 / 1.00 |
| `no-memory+trace-1-0.8+actor-0.1` (signal, cue agreement) | 0.30 / 0.01 / 0.10 | 0.31 / 0.11 / 0.20 |
| `frozen-actor` (signal, cue agreement) | 0.87 / 0.96 / 0.97 | 0.86 / 0.97 / 0.91 |
| `defaults` (noise, dependence) | 0.00 / 0.00 / 0.00 | 0.02 / 0.00 / 0.00 |
| `trace-1-0.8+actor-0.1` (noise, dependence) | 0.07 / 0.00 / 0.00 | 0.11 / 0.02 / 0.00 |
| `trace-1-0.8+actor-0.03` (noise, dependence) | 0.06 / 0.01 / 0.01 | 0.12 / 0.04 / 0.01 |

A long night at the actor rate of 0.1 erodes a 144-decision day (1.00 to 0.88);
at 0.03 the same night keeps it (0.93 to 1.00). The smaller the rate, the safer a
long night, and ten rounds of a short day are no worse than three.

## The working trace on a continuing bandit

The chamber's second instrument is the contextual bandit of
`tests/test_generic.py` (four one-hot contexts, the matching action pays one) run as
one continuing life: one stream, no resets, 600 decisions through `step` with
`gamma=0, lam=0`, the greedy hit rate on 200 fresh contexts read from a frozen
copy afterwards (chance 0.25). `Brain.build(4, 4)` of release 0.84.3, which has no working trace,
reached 0.94, 1.00 and 1.00 over three seeds; `Brain.compose(4, 4)` at its defaults
0.33, 0.21 and 0.29. The grid below varies the composed trace and the actor rate.

| Working trace (amplitude, decay) | Actor rate 1.0: hit rate per seed | Actor rate 0.1: hit rate per seed |
| --- | --- | --- |
| 3.0, 0.2 (compose default) | 0.24, 0.22, 0.30 | 0.60, 0.66, 0.72 |
| 1.0, 0.2 | 0.23, 0.21, 0.26 | 1.00, 0.95, 0.70 |
| 1.0, 0.8 | 0.27, 0.23, 0.32 | 0.96, 0.85, 0.98 |
| 0.5, 0.8 | 0.67, 0.33, 0.48 | 1.00, 1.00, 1.00 |
| 0.3, 0.8 | 0.72, 0.95, 1.00 | 1.00, 1.00, 1.00 |
| 0.1, 0.8 | 1.00, 1.00, 1.00 | 1.00, 1.00, 1.00 |
| 0.0 (trace off) | 1.00, 1.00, 1.00 | 1.00, 1.00, 1.00 |

Where the previous moment carries nothing the decision needs, the trace of the
association cortex, at amplitude 3.0 into the prefrontal projection of scale 12, holds
that cortex on its own history, and the actor at rate 1.0 cannot learn it away. The
same brain with `gamma=0.9, lam=0.8` gives the same picture (that build 1.00, 0.90,
1.00; the composed defaults 0.23, 0.28, 0.21; the trace off 1.00, 0.90, 1.00). This is
one task; the [vanished-cue chamber](../recall/README.md) needs the trace the bandit
does not, and neither selects a default.

## Limits

One synthetic world with a two-valued cue, one composition, five seeds, no physical
time. The readings are of a frozen copy on the day's own observations; held-out
days are not scored. The bandit grid has three seeds. These numbers describe the
measured settings and nothing beyond them; an application must measure its own
day, its own night and its own awake control before adopting a slept brain.
