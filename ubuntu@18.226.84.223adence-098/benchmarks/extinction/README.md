# The bandit of issue 210: a reversal whose failures pay nothing

This chamber measures `memory_extinction`
([#210](https://github.com/muellerberndt/cadence/issues/210)) against the episodic choice
without it. It establishes no default.

## The task

One constant observation and two arms. Arm 0 pays 1.0 for 120 trials, then arm 1 for 120
trials, then arm 0 again for 120; the arm that does not pay pays `--fails` (0.0 or -1.0). One
continuing `Brain.live` stream per seed: `Brain.compose(inputs=1, actions=2, modules=(16,),
seed=seed, arousal=True, memory_choice=True)`, with `memory_extinction=theta` beside it where
`--thetas` is above 0.

A failure that pays 0 writes nothing to the episodic records, so without the rule the record
the first arm earned keeps voting for it after the reversal. With the rule a zero outcome writes
a learned zero for a record the chosen action earned by its own payments in this place. Where
failures pay -1 no outcome is zero, and the rule never acts.

## Readings

Per life and phase: the hit rate of the phase's last 40 trials (a phase is acquired at 0.8 or
more), the reversal lag (the first trial from which 20 executed actions in a row include at least
18 of the paying arm; a phase never acquired counts its 120 trials), the place record of each
arm at the phase's end, the share of aroused trials, and in the later phases the aroused
failures of the arm that paid before and how many of them were written.

## Run

From the repository root:

```sh
PYTHONPATH=src python benchmarks/extinction/bandit.py --thetas 0 0.5 --fails 0 -1 \
    --seeds 0-11 11001-11012 --out DIR/bandit.jsonl --workers 4
python benchmarks/extinction/bandit.py --summary DIR/bandit.jsonl
python -m pytest -q benchmarks/extinction
```

A life takes a few seconds on one core.

## Readings on 0.93

Twelve development seeds (0 to 11) and twelve confirmation seeds (11001 to 11012, a panel
declared before it ran), read on the 0.93.0 release; the confirmation lives repeat the panel's
byte for byte. Lives acquiring the reversal (phase B), its median lag, and the median lag of the
return to arm 0 (phase A again; every life acquires it):

| Seeds | Failures pay | Without the rule | With `memory_extinction=0.5` |
| --- | --- | --- | --- |
| 0 to 11 | 0 | B on 5 of 12 (lag 120), A again lag 11 | B on 11 of 12 (lag 64), A again lag 15.5 |
| 11001 to 11012 | 0 | B on 3 of 12 (lag 120), A again lag 6 | B on 11 of 12 (lag 71.5), A again lag 15 |
| both | -1 | B on 24 of 24 | the same lives, byte for byte |

Without the rule the first arm's record ends phase B at 0.92 to 0.93 of what it was; with it at
0.57 to 0.63. The return to the first arm is slower with the rule (median 15 trials against 6 on
the confirmation seeds): its record was extinguished while the second arm paid.
