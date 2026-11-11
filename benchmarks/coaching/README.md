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

What the readings show, at three founders:

- The population code turns a brain that balances no start into one that balances some on every
  founder, and holds the pole about seven times as long on average (374 steps against 53).
- A dense block that reads the same at every moment undoes most of that.
- The first round on the brain's own path drops every founder to 0 starts and under 80 steps;
  coaching alone recovers two founders of three to 32 of 32 by the fourth round. Mixed with watched
  episodes no founder falls below 259 steps, and one of three reaches 32 of 32 in four rounds.
- With the efference copy one founder of three stays at 0 of 32 through the four rounds.

Three founders separate the code from the raw readings and the collapse from its absence; they do
not rank `coach` against `mixed` at the fourth round.

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

Six founders ([receipt](results/imbalance.json)). Weighting moves agreement from the commonest
answer to the rarest: inverse weights raise FIRE on the two weakest founders from 0.74 and 0.83 to
0.92 and 0.93 and lower LEFT; unweighted lessons agree best overall. A watched brain here does not
collapse onto the commonest answer, so the weighting is no general remedy; read each answer's
agreement and weigh only where the rare answer is the one that matters.

## Which opt-ins a coached brain needs

[`ablation.py`](ablation.py): the cart-pole task and body; a brain of modules `(64, 32)` watches
eight demonstrated episodes, then is taught on its own path only, three rounds of 16 episodes,
every moment taught before its act (``full-after``: after it). The arms are the set `arousal`, `maturity`,
`memory_fade=1.2e-4`, `memory_choice`, `efference_amplitude=1.0` and `learning_compiled_moment`,
the set less each in turn, and the default brain.

```sh
cd benchmarks/coaching && PYTHONPATH=../../src python ablation.py --out results/ablation.json
```

Starts balanced for 500 steps of 32, after watching and after each round ([receipt](results/ablation.json)):

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
reads faster after the first round. Taught after each act rather than before it, the whole set
reads about the same: the order the library recommends is not what decides this chamber.
