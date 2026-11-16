# Coaching chambers: what a brain sees, and whose path its lessons follow

[`cartpole.py`](cartpole.py) reproduces Gymnasium's CartPole-v1 in NumPy (the same equations,
constants, Euler step, termination at 2.4 m or 12 degrees, and the 500-step limit), so it needs
nothing beyond the library. A demonstrator pushes toward where the pole falls, with small position
and velocity terms against drift (`theta + 0.5 theta_dot + 0.05 x + 0.2 x_dot > 0`), and balances
every one of the 32 held-out starts for 500 steps.

Each arm composes a default brain per founder, watches eight demonstrated episodes (taught before
each act, as `step(..., teacher=labels)` orders them) and reads the 32 held-out starts frozen:
greedy acts on a copy, nothing taught.

```sh
PYTHONPATH=src python benchmarks/coaching/cartpole.py --out benchmarks/coaching/results/cartpole.json
```

| Arm | What changes |
| --- | --- |
| `raw` | the four readings as they come |
| `code` | the readings through `PopulationCode` (16 tuned inputs a reading) |
| `code+dense` | as `code`, beside 40 inputs that read 0.75 at every moment |
| `coach` | `code`, then four rounds of 16 episodes on the brain's own path, every moment taught |
| `mixed` | as `coach`, each of 8 own episodes a round followed by a watched one |
| `coach+efference` | as `coach`, on a brain with `efference_amplitude=1.0` |
| `percentile` | as `code`, over ranges from the 1st to the 99th percentile of each reading in eight other demonstrated episodes, `clamp=False`: a reading past its range fades and goes dark |
| `percentile+clamp` | the same ranges, a reading past its range read at the nearer end (`clamp=True`, the code's default) |

## Readings

Founders 0, 1 and 2 (Cadence 0.97.0 with this chamber, [receipt](results/cartpole.json)):
starts balanced for 500 steps of 32, and mean steps, after watching and after each round.

| Arm | Founder 0 | Founder 1 | Founder 2 |
| --- | --- | --- | --- |
| `raw` | 0/32, 78.0 | 0/32, 31.3 | 0/32, 48.8 |
| `code` | 22/32, 475.5 | 1/32, 322.9 | 7/32, 324.1 |
| `code+dense` | 0/32, 236.0 | 0/32, 183.4 | 0/32, 296.7 |
| `coach` rounds 1 to 4 | 0, 9.7; 0, 332.1; 0, 411.9; 32, 500.0 | 0, 75.5; 5, 321.4; 0, 269.2; 0, 326.4 | 0, 16.9; 0, 242.5; 32, 500.0; 32, 500.0 |
| `mixed` rounds 1 to 4 | 7, 404.8; 10, 403.5; 0, 259.2; 10, 453.1 | 14, 459.0; 6, 393.6; 13, 458.4; 32, 500.0 | 10, 385.0; 0, 263.7; 14, 464.7; 21, 367.3 |
| `coach+efference`, watched then rounds 1 to 4 | 23, 459.1; 0, 342.7; 9, 438.6; 6, 425.9; 8, 409.9 | 31, 498.5; 29, 461.1; 1, 191.8; 2, 424.8; 28, 496.9 | 9, 418.6; 0, 226.7; 0, 171.2; 0, 285.3; 0, 309.3 |
| `percentile` | 1/32, 92.5 | 0/32, 67.5 | 2/32, 89.1 |
| `percentile+clamp` | 1/32, 133.9 | 1/32, 124.9 | 3/32, 172.2 |

What the readings show, at three founders:

- The population code turns a brain that balances no start into one that balances some on every
  founder, and holds the pole about seven times as long on average (374 steps against 53).
- A dense block that reads the same at every moment undoes most of that.
- The first round on the brain's own path drops every founder to 0 starts and under 80 steps;
  coaching alone recovers two founders of three to 32 of 32 by the fourth round. Mixed with watched
  episodes no founder falls below 259 steps, and one of three reaches 32 of 32 in four rounds.
- With the efference copy one founder of three stays at 0 of 32 through the four rounds.
- Ranges fitted to the demonstrator's own states hold the pole 83 steps on average, and 144 with
  a reading past its range clamped to the nearer end, against 374 for the ranges of the task's
  limits. Those percentiles span 0.031 rad of pole angle and 0.62 m of cart position, against
  limits of 0.21 rad and 2.4 m; a brain that errs leaves the demonstrator's states, and past a
  narrow range every state reads alike (clamped) or not at all.

Three founders separate the code from the raw readings and the collapse from its absence; they do
not rank `coach` against `mixed` at the fourth round.

## Corrections beside watched episodes, and many streams

Four more arms of [`cartpole.py`](cartpole.py) teach on the brain's own path, read on twelve
founders beside `coach` and `mixed` (Cadence 0.98.0 on an x86-64 machine,
[receipt](results/own-path.json)):

```sh
cd benchmarks/coaching && PYTHONPATH=../../src python cartpole.py \
    --arms coach mixed streams streams-own correct correct+rehearsal \
    --founders 0 1 2 3 4 5 6 7 8 9 10 11 --out results/own-path.json
```

- `correct`: each of the brain's own episodes is played without lessons, then taught as one
  correction through `cadence.speech.correct.Rehearsal(rows=0)`;
- `correct+rehearsal`: the correction heard beside three of the eight watched episodes at lesson
  weight 0.25 (`Rehearsal(rows=3, weight=0.25)`);
- `streams`: 32 poles as the brain's 32 streams, a lesson of 32 rows a step; 125 watched steps
  (4,000 rows), then rounds of 250 steps with 16 poles moved by the brain and 16 by the
  demonstrator, a fallen pole replaced on its stream;
- `streams-own`: as `streams`, every pole moved by the brain in the rounds.

Each cell: founders of twelve at 32 of 32 starts, the mean steps over the founders, and the
lowest founder's mean steps.

| Arm | After watching | Round 1 | Round 2 | Round 3 | Round 4 |
| --- | --- | --- | --- | --- | --- |
| `coach` | 0, 315, 102 | 0, 231, 11 | 2, 314, 39 | 6, 361, 13 | 6, 418, 12 |
| `mixed` | 0, 315, 102 | 0, 372, 212 | 1, 357, 63 | 1, 385, 89 | 3, 409, 248 |
| `correct` | 0, 315, 102 | 1, 273, 55 | 3, 400, 247 | 4, 412, 212 | 4, 407, 210 |
| `correct+rehearsal` | 0, 315, 102 | 10, 460, 28 | 11, 494, 431 | 9, 474, 282 | 11, 487, 346 |
| `streams` | 0, 231, 10 | 7, 485, 352 | 10, 494, 434 | 7, 446, 54 | 9, 489, 391 |
| `streams-own` | 0, 231, 10 | 7, 488, 430 | 11, 492, 407 | 10, 477, 289 | 9, 473, 233 |

- Corrections heard beside watched episodes hold the gain from the first round: ten founders of
  twelve balance every start after it, against one with the correction alone and none for `coach`
  and `mixed`.
- Many streams reach the same level by the second round, with every pole on the brain's own path
  (`streams-own`) as with half of them.
- The single-stream arms `coach`, `mixed` and `correct` end with three to six founders of twelve at
  32 of 32 after four rounds.

Founders 0 to 2 of `coach` and `mixed` read differently here than in the three-founder table
above, which was read on Cadence 0.97.0 on Apple silicon.

The same twelve founders after watching, with the wide code (`code`) beside a narrow one of the
demonstrator's states (`code+narrow`, [receipt](results/narrow.json)): 235 steps on
average against 315 for the wide code alone and 165 for the narrow one alone.

## The scan-and-fire chamber

[`imbalance.py`](imbalance.py): a field of 32 columns shows a target as a bump around its column,
or nothing. The demonstrator scans LEFT while nothing is in view, turns toward a target off centre
and FIREs within two columns of the centre; a turn shifts the view three columns and a shot clears
the target. LEFT covers the scan and the turn left, 73% of the held-out moments; FIRE is 11% and
RIGHT 16%. A default brain per founder watches 40 episodes of 40 moments and is read frozen on 20
held-out episodes, per answer.

```sh
PYTHONPATH=src python benchmarks/coaching/imbalance.py --out benchmarks/coaching/results/imbalance.json
```

| Lessons | Overall | LEFT | FIRE | RIGHT |
| --- | --- | --- | --- | --- |
| unweighted | 0.90 to 0.98 | 0.91 to 1.00 | 0.74 to 0.98 | 0.94 to 0.98 |
| inverse of the share | 0.86 to 0.96 | 0.83 to 0.96 | 0.86 to 1.00 | 0.94 to 1.00 |
| inverse square root | 0.79 to 0.98 | 0.72 to 0.99 | 0.84 to 1.00 | 0.96 to 1.00 |
| every row (`learning_surprise=False`) | 0.97 to 0.99 | 0.99 to 1.00 | 0.87 to 0.98 | 0.88 to 0.99 |

Six founders ([receipt](results/imbalance.json)). Weighting moves agreement from the commonest
answer to the rarest: inverse weights raise FIRE on the two weakest founders from 0.74 and 0.83 to
0.92 and 0.93 and lower LEFT. The first three rows teach only the rows the brain answers wrong
(`learning_surprise=True`, a new brain's default); the fourth teaches every row at weight 1 and
agrees best overall: the weakest founder of the unweighted lessons rises from 0.90 overall and 0.74
on FIRE to 0.97 and 0.87, while one founder's RIGHT falls from 0.98 to 0.88. A watched brain here
does not collapse onto the commonest answer, so the weighting is no general remedy; read each
answer's agreement and weigh only where the rare answer is the one that matters.

## Which opt-ins a coached brain needs

[`ablation.py`](ablation.py): the cart-pole task and body; a brain of modules `(64, 32)` watches
eight demonstrated episodes, then is taught on its own path only, three rounds of 16 episodes,
every moment taught before its act (``full-after``: after it). The arms are the set `arousal`, `maturity`,
`memory_fade=1.2e-4`, `memory_choice`, `efference_amplitude=1.0` and `learning_compiled_moment`,
the set less each in turn, and the default brain.

```sh
cd benchmarks/coaching && PYTHONPATH=../../src python ablation.py --out results/ablation.json
```

Starts balanced for 500 steps of 32, after watching and after each round ([receipt](results/ablation.json);
read again with `clamp=True`, the code's default since 0.97.1, every row is the same to the bit):

| Arm | Founder 0 | Founder 1 | Founder 2 |
| --- | --- | --- | --- |
| the whole set | 0, 23, 30, 32 | 2, 4, 14, 32 | 0, 0, 32, 32 |
| less `arousal`, `maturity` or `memory_fade` | the same, to the bit | | |
| less `memory_choice` | 11, 27, 2, 21 | 0, 17, 32, 32 | 0, 26, 10, 32 |
| less `efference_amplitude` | 6, 32, 32, 32 | 0, 13, 32, 32 | 2, 21, 32, 32 |
| less `learning_compiled_moment` | 0, 32, 32, 32 | 2, 24, 32, 32 | 0, 32, 32, 32 |
| the default brain | 0, 4, 10, 14 | 23, 16, 8, 15 | 0, 3, 0, 24 |
| the whole set, taught after each act (`full-after`) | 5, 32, 32, 32 | 7, 10, 32, 32 | 2, 0, 32, 32 |

Arousal, maturity and fading records leave lessons unchanged, so those arms repeat the whole set.
The efference copy slows the first round; without the episodic memory the rounds swing. The
compiled moment computes the same equations in another order (other bits), and the arm without it
reads faster after the first round. Taught after each act, not before it, the whole set
reads about the same: the order the library recommends is not what decides this chamber.
