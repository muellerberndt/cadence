# Scaling of the recall: mastery with training, width and memory

Declared 17 October 2026, before the first run of the grid. Instrument: `scaling.py` (the grid,
the runner, the summary) on the lives of `index_runs.py`, whose parameters it sets; contracts:
`test_scaling.py`. Companion to [the index memory](INDEX_PROTOCOL.md), whose `horizon` arm
(amendment 5) is the recall read here.

## Question

The index protocol gates held-out answers at .75 on single tasks at one size. Does the recall
help the composed brain reach mastery, and does mastery come with enough training, a large
enough brain and a large enough memory? On a declared family of memory-dependent tasks, each
with a difficulty knob, the readings support or refute the sentence: with the recall on, a
capacity that holds the task's span, and enough width and training, the brain masters every
level of every task of this family; without the recall, at the same width and training, it does
not.

## Task family

Every task is an instrument of this directory, read through `index_runs`' lives with
parameters that take the declared task's value when absent, so every declared task reads as
before (a contract test checks the token panels; the declared lives' readings are unchanged).

| Family | Instrument | Knob: levels | Fixed at every level | Held-out measure (chance) |
| --- | --- | --- | --- | --- |
| `token` | `token_binding`'s trials | facts per trial F: 2, 4, 8; neutral events between the facts and the query D: 1, 8, 32 (nine levels) | nine name tokens, eight trained and one never, and eight values (eight facts need eight distinct names); the instrument's held-out rule: one value per training name, each value held out for exactly one name | accuracy on 128 held-out trials at the level's F and D, the queried fact a held-out pair (1/8) |
| `succession` | `sequence_binding`'s trials | tokens per sequence L: 4, 8, 16 | seventeen tokens, sixteen trained and one never; one held-out successor per training token, a permutation without fixed points | accuracy on 128 held-out sequences of length L, the cued transition the sequence's only held-out one (1/16) |
| `named` | `named_bindings` through the fresh runner | bits per fact: 3, 4, 5 | the declared encoding: both facts on shared coordinates, a query that shares nothing with either; a quarter of the value pairs held out | exact match over every held-out pair, both orders, both queries (2^-bits) |
| `continuing` | token binding without a reset between trials | earlier trials of the same life before each panel trial K: 2, 8, 32 | the declared continuing task: five name tokens, two facts, one neutral event | accuracy on 128 held-out trials, each read after K earlier trials (1/8) |
| `camera` | token binding at 12,824 inputs through the generic code | code units: 256, 1024, 4096 | winners a tenth of the units (26, 102, 410); fan-in 32, raised by the cover rule to 200 at 256 units and 50 at 1024; the camera task's world, vocabulary and `heard` 1.5 | accuracy on the frames of 128 held-out trials (1/8) |

Each level trains at its own setting: training trials at the level's F and D, L or bits, and
the continuing life's slots hold what its capacity holds. The token level F 2, D 1 differs
from the declared token task in its vocabulary alone; the camera level of 1024 units differs
from amendment 2's generic code in its winners (102 for 96).

## Factors

| Factor | Settings |
| --- | --- |
| memory | `none`: the composed brain as the instrument builds it without a read (`IndexBrain.build(read=None)`: sensory, association, its prefrontal working trace, motor; no recall or next region), its working trace the only carrier of the facts. `recall`: the index memory of the `horizon` arm, the `index` genes with `recency` None (the charge `r log(1 + age)`, `r = heard^2 / T / log(1 + capacity)`) and the capacity below |
| association width | 64, 256, 1024: the association region and its working trace (the prefrontal region holds one neuron per association neuron). The instrument sets 64 for every task (`token_binding.WIDTH`, the fresh runner's m64). The recall and next regions keep the observation's width; the address map's cortex part grows with the width |
| capacity (recall) | 16, 64, 256 where the task's span needs it: the token levels of D 32 (their spans, 34 to 40 moments, exceed 16) and the continuing life (the panel holds K earlier trials of four events when the capacity is at least 4K + 3: K 2 at every capacity, K 8 at 64 and 256, K 32 at 256; a smaller capacity drops the oldest); 64 elsewhere, which holds every other level's span (at most 16 moments) |
| training | 256, 1024, 4096 and 16384 rounds of 16 trials, one lesson per round at the query, judged by the act (`learning_surprise`), all in one life read at each budget on forks that never change it |
| founders | 0, 1 and 2 (development founders) |

## Readings

Per life and budget, on the held-out panel: intact; with the slots erased before the query
(recall); with the working trace erased before the query; for `named` also both erased. On the
training panel (128 trials of the training distribution; for `named`, every training pair):
intact. The erased conditions say which carrier holds the facts; the training panel says what a
table over trained patterns reaches.

## Mastery

A founder is at mastery at a budget when its held-out measure is at least .95. For each
setting (level, memory, width, capacity) the summary reports, at every budget, the mean
held-out measure over the founders and the founders at mastery; the first budget at which every
founder is at mastery, or none; and for `recall` against `none` at the same level, width and
budget the paired difference, recall less none, per founder and in the mean.

A life ends once two consecutive readings are at mastery; its later budgets are not read, and
the summary carries its last reading to them and marks the mean that counts a carried reading.
The rule assumes a mastery held over a fourfold extension of training stays held. Without
mastery a life reads every budget of its setting.

## Grid

The full grid (`--grid full`): 21 levels; at each level and width, `none` and `recall` at every
capacity of the level; three widths, four budgets in one life, three founders: 486 lives (token
216, succession 54, named 54, continuing 108, camera 54). A training round's processor time
grows with the width and with the events per trial (41 at F 8, D 32). Camera brains at width
1024 with 1024 or more code units hold more than the library's dense limit of 2048^2 block
entries, so every settle takes the slower sparse path. The full grid does not fit one box.

The declared grid (`--grid trimmed`, `scaling.py`'s `TRIMMED`): 324 lives. Every level runs at
widths 64 and 256; width 1024 runs at the easiest token level, where a wide brain without the
memory has its best chance, and at the hardest levels of every family but the camera's (whose
brains take the sparse path at that width), to the budgets their lives allow. A cell
names the last budget read (every budget up to it is read) and the recall's capacities where
the factor applies; the brain without the memory reads the cell's longest budget.

| Family | Level | Width 64 | Width 256 | Width 1024 |
| --- | --- | --- | --- | --- |
| `token` | F 2, D 1 | 16384 | 16384 | 16384 |
| `token` | F 4 or 8, D 1; F 2 or 4, D 8 | 16384 | 16384 | |
| `token` | F 8, D 8 | 16384 | 16384 | 4096 |
| `token` | F 2 or 4, D 32 | 16384 at capacities 64 and 256, 4096 at 16 | 16384 at 64 | |
| `token` | F 8, D 32 | 16384 at capacities 64 and 256, 4096 at 16 | 16384 at 64 | 1024 at 64 |
| `succession` | L 4 or 8 | 16384 | 16384 | |
| `succession` | L 16 | 16384 | 16384 | 4096 |
| `named` | 3 or 4 bits | 16384 | 16384 | |
| `named` | 5 bits | 16384 | 16384 | 16384 |
| `continuing` | K 2 | 16384 at 16, 64 and 256 | 16384 at 16 | |
| `continuing` | K 8 | 16384 at 16, 64 and 256 | 16384 at 64 | |
| `continuing` | K 32 | 16384 at 16, 64 and 256 | 16384 at 256 | 4096 at 256 |
| `camera` | 256 or 1024 units | 16384 | 16384 | |
| `camera` | 4096 units | 4096 | 1024 | |

Capacity 16 at D 32 reads to 4096: its slots cannot hold the facts. The grid was chosen from
processor seconds per round and per reading of untrained founders at small settings
(`scaling.py probe`); no score of any task entered it. A setting read to a shorter budget says
nothing about the budgets it does not read, and a width or capacity left out says nothing about
that width or capacity. `--grid smoke` (budgets 2 and 4) tests the pipeline and reads nothing.

## Instrument

The parameters enter the instruments as arguments whose defaults are the declared values:
`token_binding` takes the number of name tokens (`holdout`, the events; `draw` reads the
training names from the held-out table), `sequence_binding` the number of tokens, and
`index_runs`' lives the parameters of the family (its module note lists them). The lives
stamp each reading with processor seconds (`cpu`, and `read_cpu` for the reading alone). At
their defaults the declared lives read as before: the readings of every declared task at two
budgets on one founder are unchanged, and a contract test checks the token panels trial for
trial.

## Declared expectations

- `recall` reaches mastery at every level once the capacity holds the span and the width and
  training suffice; the budget at mastery grows with F, D, L, bits and K. With the slots
  erased its held-out measure falls to chance; with the trace erased it stays near intact.
- `recall` at capacity 16 at D 32 stays below mastery: the facts leave the slots before the
  query. In the continuing life a capacity larger than the panel's history adds earlier facts
  with the same names, and the power-law charge, weaker at a larger capacity, must still favor
  the latest; K 32 asks whether it keeps the current trial's fact ahead of up to 32 earlier
  ones, and a failure there with mastery at K 2 and 8 places the limit in the charge.
- `named` needs the address lesson to learn the route from an ask marker to the named fact;
  more bits need more rounds.
- `none` plateaus below mastery on the held-out measure at every level and width: its working
  trace answers from a table over trained patterns (as the token protocol expects of its graph
  learner on held-out pairs), and its training-panel score may rise with width and training
  while the held-out score does not. Erasing its trace brings its held-out score
  to chance.
- `camera`: the recall reaches mastery at every code width; wider codes separate the tokens'
  codes further and cost more per act.

## What a pass would and would not establish

A pass, every level of every family at mastery on all three founders for `recall` at some
declared width, capacity and budget, with `none` below mastery at the same width and budget,
supports the sentence of the question for this family, on these founders and budgets. It does
not establish mastery of every task: the family is five instruments with bounded knobs (at most
eight facts, 32 neutral events, sixteen tokens, five bits, 32 earlier trials, 4096 code units),
one-hot or coded events, lessons at the answer alone. It establishes no chained recall (the
index protocol expects that below its gate), no confirmation (the founders are development
founders; confirmation founders stay unopened), no runtime default and nothing about C64
transcription, spoken words or a world's tasks, which wait on their own instruments. A failure
at a level says which part did not suffice where the readings separate them: the capacity (the
slots erased change nothing), the width (a wider setting masters), the training (the mean still
rising at the last budget) or the recall itself.

## Commands

From the repository root, with Cadence on the Python path and one BLAS thread per process;
`DIR` lies outside the repository. The run resumes: a second call skips every life whose
receipt is in `DIR/lives`.

```sh
PYTHONPATH=src python benchmarks/binding/scaling.py plan --grid trimmed --out DIR/jobs.json
PYTHONPATH=src python benchmarks/binding/scaling.py run --jobs DIR/jobs.json --out DIR \
  --workers 2
PYTHONPATH=src python benchmarks/binding/scaling.py summary --out DIR
python -m pytest -q benchmarks/binding/test_scaling.py
```

On a box with many cores, `bash benchmarks/binding/run_scaling.sh DIR` runs the three steps
with a worker per core; with `PROBE=1` it first measures the grid's cost classes on that box
and the plan prints the estimate before the lives start.

## Amendment 1, the linear charge beside the power-law charge (17 October 2026)

Declared while the declared grid runs, before any life of this amendment runs. The
power-law charge `r log(1 + age)`, `r = heard^2 / T / log(1 + capacity)`, is calibrated on a
full match of a unit-length key. At the camera task's `heard` 1.5 and capacity 64, `r` is 2.16:
a trial's two facts, at ages 1 and 2 when the query reads, cost 1.5 and 2.4 logits, and the
neutral event at age 0 costs nothing. A camera query matches its fact's code only in part,
so an unmatched young moment can outweigh the older fact. The grid's `recall` on the camera
family therefore reads the charge as well as the memory's width; its lives run and are
read as declared, with this amendment's beside them.

| Item | Declared |
| --- | --- |
| memory `recall_linear` | the index protocol's `index` arm, its confirmed recall: the `index` genes with `recency` 0.25 (each slot's logit lowered by 0.25 per moment of age) and the task's own genes (`heard` 1.5 in the camera task), at the capacity below |
| capacity | 64 in the token and camera levels: every trial starts from empty slots, its span fits, no slot is dropped and the linear charge does not read the capacity, so 64 stands for every capacity that holds the span, the index arm's unbounded one included. In the continuing life the smallest capacity of the factor that holds the panel's history, 64 at K 8 and 256 at K 32, so that the charge and not the capacity bounds what the read reaches |
| lives | (a) the camera family at every level, width and budget the declared grid reads: 256 and 1024 units at widths 64 and 256 to 16384 rounds, 4096 units to 4096 rounds at width 64 and to 1024 at width 256; (b) at width 64 and to 16384 rounds, the token levels of D 8 and D 32 (F 2, 4 and 8) and the continuing levels of K 8 and K 32, where the linear charge's horizon is the question |
| founders, readings, mastery | as declared: founders 0 to 2; the held-out measure intact, with the slots erased and with the trace erased, and the training panel; mastery at .95; a life ends after two consecutive readings at mastery |
| grid | `--grid supplement`: 42 lives (camera 18, token 18, continuing 6), run after the declared grid into the same output with a job list and a log of their own; their names carry the setting, so the runner's resume neither skips nor reruns a declared life |
| summary | the three memories side by side at each level and width the supplement reads (`none`, `recall` at the linear setting's capacity, `recall_linear`), and the paired differences by founder `recall_linear` less `none` and `recall_linear` less `recall` at the same capacity, beside `recall` less `none` |

### Declared expectations

- `camera`: `recall_linear` reaches mastery at the code widths the grid reads, where `recall`
  stays far below it: at ages 1 and 2 the linear charge costs 0.25 and 0.5 logits, less than a
  fact's lead of about 1.7 logits over an unmatched moment. Where both fail, the code limits.
- `token`: a match of the queried name is worth `heard^2 / T`, 4 logits, which the linear
  charge spends by age 16. At D 8 the facts sit at ages 8 to 15 when the query reads, inside
  that horizon and near its edge at F 8; at D 32 at ages 32 to 39, charged 8 to 9.75 logits.
  There the name keeps the queried fact ahead of the other facts, but the neutral moments take
  most of the read, and mastery depends on whether the address lesson's lift of the facts and
  the cortex's read of a small share suffice. A failure at D 32 with mastery at D 8 places the
  linear charge's horizon between them.
- `continuing`: `recall_linear` holds K 8 and K 32 at least as well as `recall`: an earlier
  fact with the queried name is at least three moments older (0.75 logits), and each earlier
  trial costs it a further logit, while the power-law charge at capacity 256 separates old
  facts by little, so more of them share its read.

### What the amendment would and would not establish

Where `recall_linear` reaches mastery and `recall` does not at the same capacity, the readings
place the difference in the charge; where neither does, in what the amendment does not vary.
The declared grid's lives, readings and expectations stand unchanged. A pass of the question's
sentence with the linear charge in place of the power-law charge holds for the levels this
amendment reads alone: the camera family, and at width 64 the token levels of D 8 and 32 and
the continuing levels of K 8 and 32.

### Commands

```sh
PYTHONPATH=src python benchmarks/binding/scaling.py plan --grid supplement \
  --out DIR/jobs-supplement.json
PYTHONPATH=src python benchmarks/binding/scaling.py run --jobs DIR/jobs-supplement.json \
  --out DIR --workers 2
PYTHONPATH=src python benchmarks/binding/scaling.py summary --out DIR
```

On the box, `bash benchmarks/binding/run_scaling.sh DIR WORKERS supplement` runs the three
steps into the declared grid's `DIR`.
