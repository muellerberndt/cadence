# A record's strength

Issue: [#336](https://github.com/muellerberndt/cadence/issues/336). A long-term record grows
strong when it comes up again and again or when it came with a large surprise, is kept for good
past a threshold, and fades away below it. The settings are
`cadence.LongTermMemory(blend=, familiar=, keep=, forget=, novelty=)` with
`write(..., surprise=)`, and `SynapticMemory(fade_keep=)`; the guide is
[docs/memory.md](../../docs/memory.md#a-records-strength-opt-in).

## The chamber

`strength.py` keeps sensory records in one store. A pattern is a unit code of 64 coordinates, a
take of it the pattern with noise (cosine about .86), its value its own label. A pattern is told
n times as n takes in a row, the store lives an idle span (`moment()`, no read), and each pattern
is then asked once from a fresh noisy take (cosine about .70); the read is right where it holds
the pattern's label. Two stores read the same draws:

- **main**: read by cosine, a take at cosine .6 or above told again (`revise=0.6`), the depth
  kept whole over these lives (`hold=inf`); every setting of this issue off.
- **strength**: the same with `blend=True, keep=8, forget=50, familiar=0.02`.

```bash
PYTHONPATH=src python benchmarks/memory_strength/strength.py --seeds 0 1 2
```

`test_strength_chamber.py` runs it in short; `tests/test_memory_strength.py` holds the gates as
unit tests (G1 to G6).

## Readings (seeds 0, 1 and 2; 20 patterns a cell; the range over the seeds)

Right after an idle span, by tellings (the dose curve, G1 and G3), `strength`:

| Idle moments | n = 1 | n = 2 | n = 4 | n = 8 | n = 16 |
| --- | --- | --- | --- | --- | --- |
| 0 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 60 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 300 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 |
| 1,000 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| 10,000 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| moments until forgotten, median | 54 | 100 | 850 | never | never |
| the law: forget x (1 + (n / 2) ** 4) | 53 | 100 | 850 | kept | kept |

`main` reads every record right (1.00, one cell .95) at every span: with room to spare a record
fades only toward its replacement, never out of the read.

One write after 10,000 idle moments (G2), `strength`:

| Told | right |
| --- | --- |
| one write, surprise 0, 1 or 3 | 0.00 |
| one write, surprise 7 | 1.00 |
| told 8 times | 1.00 |

The record against its takes (G4):

| Takes | `strength`: the record's cosine to the pattern | `main` | the nearest take's |
| --- | --- | --- | --- |
| 1 | .85 to .86 | .85 to .86 | .85 to .86 |
| 4 | .96 | .85 to .87 | .88 to .89 |
| 16 | .99 | .85 to .86 | .90 |

- **The law holds to the moment.** A record told n times and never read fades by
  `1 / (1 + (n / 2) ** 4)` a moment, so it is forgotten after `forget * (1 + (n / 2) ** 4)`
  moments; at `keep=8` eight tellings, or one write at surprise 7, keep it for good.
- **A told-again take on main replaces the record**: its cosine to the pattern stays that of one
  take however often it is told. With `blend` the record is the mean of its takes.
- **What is designed, by name:** the threshold, the forgetting level, the fourth-power law of the
  fade (the fade of 0.99.0), the mean of the takes and the weight on the read. Nothing here is
  learned; the surprise is the caller's (or the store's distance to its nearest record).
