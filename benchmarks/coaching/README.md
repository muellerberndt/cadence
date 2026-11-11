# The cart-pole chamber: what a brain sees, and whose path the lessons follow

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
