# Sampled-income repair in saved robot lives

Development comparison for [#158](https://github.com/muellerberndt/cadence/issues/158),
2026-10-08. This checks the income repair on continuing acquired robot brains;
it does not establish improved fighting skill. The related exploration and greedy
policy limitations in #159/#160 were outside this income-only comparison.
The subsequent [policy repairs and their mixed arena results](../arena_policy/README.md)
are recorded separately; the broader #111 capability gate remains open.

The arena is [cadence-robot-arena](https://github.com/muellerberndt/cadence-robot-arena)
at `14eb800543923c9e1f64f28319490ea4435decb6`. Each run loads the same six evolved
checkpoints and the previous fight's actual owed reward and terminal flag from
`league.json`. The first moment delivers that outcome to the checkpoint's pending
action, exactly as `League.royale` does. Terminal feedback can clear short-term
traces under the normal lifecycle; there is no forced brain or arousal reset.
The declared `need=0` control isolates relative-income want, and heat/temperature
remain the saved genes.
The original positive-need condition can legitimately remain aroused when its
actual income does not meet its need.

Each arm runs three 600-moment royales (seeds 77, 78, 79; closing zone 500),
with no saving and no replay. Each run starts from the original acquired
checkpoints, rather than claiming a sequence of three continued fights.
The third arm uses uniform-random commands with the same six bodies. Cadence
control is `2566ccf2c23b76505a82b941111535ffd50f8ae4`; candidate is `bf90fa4`,
with the same runtime change as `9c2ac1f`. The JSON retains source identities,
the arousal module and harness hashes, every checkpoint hash, initial and final
owed outcomes, per-fighter readings and work. The final terminal outcome is
recorded as still owed, not delivered after the run. All checkpoint and league
hashes are unchanged after each arm.

| Measured quantity | Source control | Income repair | Uniform random |
| --- | ---: | ---: | ---: |
| Total lived moments | 10,533 | 10,575 | 10,800 |
| Aroused share, weighted by lived moments | 92.8% | 65.8% | n/a |
| Moving closer to nearest rival, weighted share | 47.6% | 36.4% | 25.0% |
| Damage dealt, all fighters | 680.1 | 587.5 | 0.0 |
| Learning sweeps | 99,025 | 73,191 | 0 |
| Refusals | 0 | 0 | 0 |

Weighted shares use the arena's rounded per-fighter readings and are approximate.
Changed survival means denominators differ. All three source-control runs produce
identical behavioral readings; all three random runs also agree. These
are repeated development runs, not independent fresh-founder confirmation.

The repair reduces time spent aroused without a reset, but closing behavior
deteriorates. There is no demonstrated improvement in fighting tactics. Returning
these existing brains to greedy operation exposes their already documented weak
greedy policies; that is consistent with #160, but this comparison does not prove
the cause. No learning rule, temperature, heat, sensory gain or acceptance gate
is tuned to improve these results.

The original positive-need condition is also compared on seed 77 with the same
corrected harness and actual owed feedback, without a stage override. Sustained
want remains legitimate when income falls below need; this check does not show
improved fighting either. Candidate arousal ranges from 71.2% to 86.3% across
the six robots, compared with 100% for every control robot.

| Original positive need, seed 77 | Source control | Income repair |
| --- | ---: | ---: |
| Total lived moments | 3,439 | 3,246 |
| Aroused share, weighted by lived moments | 100.0% | 78.9% |
| Moving closer to nearest rival, weighted share | 44.8% | 40.7% |
| Damage dealt, all fighters | 294.7 | 318.4 |
| Learning sweeps | 36,213 | 27,493 |
| Refusals | 0 | 0 |

Damage rises in this one paired run while closing and survival time decline.
It is a development observation, not evidence of consistently better tactics.

The [law regression](../../tests/test_arousal_income.py) isolates the actual bug:
after a reward drop, 1,500 non-greedy sampled outcomes leave the old income
reference frozen and the final 300 outcomes all aroused. The candidate follows
the new income and is calm on all final 300 outcomes, with own-error calibration
unchanged. [Paired nursery preservation](../reversal/README.md#income-tracking-repair--2026-10-08)
separately retains the original capability gates and records increased reversal
lags and learning cost.

## Superseded readings without owed feedback

The first harness and the arena's unchanged `scripts/arousal_readings.py` omitted
the terminal outcome stored separately in `league.json`. Their first moment
therefore supplied the pending action a zero reward with `done=False`; loading
the checkpoint alone did not preserve actual-outcome custody. Those readings are
retained byte-for-byte in `results/without-owed/`, together with the original
harness. They describe a perturbed continuation and are superseded by the
corrected comparison above.

In that limited first comparison, weighted arousal fell 93.4% to 66.2% and
closing behavior fell 38.2% to 27.7%, against 25.0% random. Its positive-need
readings were 100% aroused under the control and 81–88% under the candidate.
Neither the original results nor the correction demonstrates acquired fighting
skill. The unfavorable readings remain part of the evidence record.

## Reproduce

The harness is retained exactly as run. Copy `compare_arena.py` to
`<arena>/runs/cadence-issues-158-160-20261008/compare_arena.py`; it resolves that
arena checkout from its location. From the arena checkout, use its environment
and select the exact Cadence checkout with `PYTHONPATH`:

```sh
PYTHONPATH=/path/to/cadence/src:/path/to/cadence-robot-arena \
  .venv/bin/python runs/cadence-issues-158-160-20261008/compare_arena.py \
  --need-zero --output runs/issue158-comparison.json
```

Repeat with the control and candidate source checkouts; add `--policy random`
for the uniform-random arm. To reproduce the positive-need check, omit
`--need-zero` and use `--seeds 77`. Output is local only. No league checkpoint is written.
The supplied files in `results/` retain both unfavorable and favorable readings.
