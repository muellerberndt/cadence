# Key-door operating point re-selected under the current credit law (2026-10-15)

Status: declared before the first run, for
[issue 198](https://github.com/muellerberndt/cadence/issues/198). The instrument is
`key_door.py`, unchanged; the grid, the seeds and the gate are in
[`protocol-3-reselection.json`](protocol-3-reselection.json). Nothing here changes
`protocol-3.json`, the runner, its tests or any default. Receipts are kept outside this
repository.

## Why

The declared operating point of [`protocol-3.json`](protocol-3.json) was selected on 0.77.0.
Two commits of 0.79.0 changed what the live brain learns:
[b041dc7](https://github.com/muellerberndt/cadence/commit/b041dc7) tracks arousal income
during sampled actions, and [50c1c97](https://github.com/muellerberndt/cadence/commit/50c1c97)
credits the policy that sampled an action at its own temperature, so an aroused update at
heat 2 is half as large and follows the hot policy. Issue 198 records the move of the declared
point's reading on the development seeds and its bisection to those two commits. Until the
point is selected again under this credit law, the chamber's reading of a gene measures the
mismatch between its point and the library.

## The grid

Eight points around the declared one, each a `--point` override of the copy's operating point:

| point | actor step | bias step (eta / 10) | critic rate | lam |
| --- | --- | --- | --- | --- |
| `eta0.1-critic5-lam0.95` (declared) | 0.1 | 0.01 | 5 | 0.95 |
| `eta0.1-critic5-lam0.8` | 0.1 | 0.01 | 5 | 0.8 |
| `eta0.1-critic1-lam0.95` | 0.1 | 0.01 | 1 | 0.95 |
| `eta0.1-critic1-lam0.8` | 0.1 | 0.01 | 1 | 0.8 |
| `eta0.03-critic5-lam0.95` | 0.03 | 0.003 | 5 | 0.95 |
| `eta0.03-critic5-lam0.8` | 0.03 | 0.003 | 5 | 0.8 |
| `eta0.03-critic1-lam0.95` | 0.03 | 0.003 | 1 | 0.95 |
| `eta0.03-critic1-lam0.8` | 0.03 | 0.003 | 1 | 0.8 |

Every other setting stays as declared: modules (32), trace amplitude 0.3, consolidation 0.25,
gamma 0.95, the arousal genes of `protocol-3.json` (need 0.03, heat 2, youth 100), the world,
the three rules of 500 trips and the gates.

## Development and selection

| step | arm | seeds | delays | lives |
| --- | --- | --- | --- | --- |
| development, each point | `live` | 0 to 11 | 2, 5 | 24 |
| confirmation seeds, the selected point | `live` | 900 to 909 | 2, 5, 10 | 30 |

Seeds 0 to 7 are the development seeds of `protocol-3.json`; 8 to 11 belong to none of its
seed sets. A life is jointly qualified when it meets all five predicates of the runner's joint
audit: acquired, adapted, frugal, calm and retained (fed on 0.9 of the last 50 trips of each
rule, at most 1.5 wrong interactions per trip, at most 0.35 of the second half of each rule
aroused, the returned key found within 50 completed trips). The selected point has the most
jointly qualified development lives of 24. A point with a crashed life is not eligible. Ties
go to the larger summed count of the five predicates (of 120), then to fewer settings changed
from the declared point, then to the larger actor step. The selected point goes to the
confirmation seeds whether or not it reaches the gate's share on the development seeds; its
development reading is reported beside it.

## The gate on the confirmation seeds

At least 0.9 of the 20 lives at delays 2 and 5 jointly qualified, none crashed; delay 10 is
reported without a gate. The runner's `admissible` and `passed` are false for every run with
`--point` by construction, so the reading comes from the receipt's `gates.pooled`
(`jointly_qualified`, `crashed`). Seeds 900 to 909 were opened by the key-door/3 freeze of
2026-10-08 and by the maintainer audits: this reading audits opened seeds under the present
library and is no fresh confirmation. A point that passes here becomes a candidate for a new
freeze with fresh confirmation seeds; the freeze is a separate declaration.

At the selected point issue 198 also compares the composed founder's actor step (1.0, bias
0.05) with the candidate founder pair (0.1, bias 0.01) on the same confirmation seeds; that
comparison belongs to the founder change and is declared with it.

## Commands

From the repository root, one worker per point, two points at a time:

```sh
python benchmarks/keydoor/key_door.py --protocol benchmarks/keydoor/protocol-3-reselection.json \
  --arms live --seeds development --delays 2 5 --workers 1 \
  --point '{"eta": 0.1, "eta_critic": 5.0, "lam": 0.95}' --out <receipt>.json.gz
python benchmarks/keydoor/key_door.py --protocol benchmarks/keydoor/protocol-3-reselection.json \
  --arms live --seeds confirmation --workers 1 --point '<selected point>' --out <receipt>.json.gz
python benchmarks/keydoor/key_door.py --verify <receipt>.json.gz --current
```

`--verify --current` checks a receipt against this copy's hash when the copy is passed as
`--protocol`, as the runner does after writing it.

## Caps

| run | cap |
| --- | --- |
| one development point (24 lives, one worker) | 90 minutes |
| the grid | 6 hours |
| the confirmation-seed reading (30 lives) | 120 minutes |

A run past its cap is stopped and reported as capped; failed and capped runs are kept and
reported as measured.

Addendum, 2026-10-15 09:41Z, after the first two points and before any run reached a cap:
other work raised the machine's load until each process of this re-selection received about a
third of a core, so the caps above apply from this time to the CPU time of each run's process
(each run is one process at `--workers 1`), with the same minutes. Points, seeds, delays, the
selection rule and the gate are unchanged.
