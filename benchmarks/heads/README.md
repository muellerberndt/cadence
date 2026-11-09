# Heads chamber: composed attention functions

Interpretability work on transformers finds functions composed across heads: one head detects a
duplicate, another suppresses attention to it, another moves the remaining name to the output.
This instrument reads that composition in one brain
([#269](https://github.com/muellerberndt/cadence/issues/269), under the attention tracker
[#272](https://github.com/muellerberndt/cadence/issues/272)): can the brain do it, where does it
fail, and how does it compare? The task that names it: `when mary and john went out john gave a
drink to` answers `mary`, with names no lesson used. Each part is read alone first, so a composed
failure can be traced to its part. Read the
[recall memory](../../docs/memory.md#the-recall-memory-in-full) and the
[grammar chamber](../grammar/README.md), whose scale and controls this chamber takes, before
interpreting results. It establishes no default.

## The language and the split

Twenty words, each an input and an action of the brain: twelve names (`mary john anna tom lisa paul
sara mark nina owen ruth sean`) and the frame `when and went out gave a drink to`. Per founder
(`Split.of`) eight names are taught and four are never-taught: never heard and never answered in
training. Four pairs of taught names are never heard together. Every life on a founder hears the
same batches, from `SeedSequence([founder, 11])`, and reads the same frozen panels, from
`SeedSequence([founder, 33])`. A batch is one shape, sixteen streams with their own content, as in
the grammar chamber.

## The tasks

Lessons fall on answer moments only, before the act on the same observation (`Brain.lesson`, then
`Brain.act`). Every other moment is heard and acted on greedily. Each task is a life of its own.
The single functions take 1,500 batches (the sequence battery's), the sentences 4,000 (one pass of
the grammar chamber).

| Task | A batch | The answer | What it reads |
| --- | --- | --- | --- |
| `previous` | a stream of 3 to 8 tokens, the last flagged `ASK` | the token before it | position over content |
| `duplicate` | 3 to 6 distinct tokens, then a token flagged `ASK` | `yes` if it was told, else `no` | the duplicate signal, no route needed |
| `successor` | 3 to 6 distinct tokens told once, each episode in its own order, then one flagged `ASK` | the token told after it | the next port by content |
| `successor_fixed` | the same with one fixed order of the taught tokens in training | the same | whether weights take over from the read |
| `induction` | `... A B ... A`, the query unflagged, 5 to 10 tokens | `B` | the sequence battery's task |
| `twohop` | pairs told, then `a`, then `THINK` | `b`, then `c` | a read on what a read found |
| `copying` | `START`, 3 to 6 distinct tokens, `GO`, each answer heard | each token in turn | closed loop (`test`) and teacher-forced (`forced`) |
| `suppression` | `A B S GO`, `S` one of the two | the one not repeated | inhibition without a frame |
| `which` | the same | `first` or `second`, the place of the one not repeated | suppression with no route needed |
| `ioi` | `when A and B (went (out)) S gave (a drink) to` | the one not repeated | the named task |
| `bare` | `A B (went (out)) S gave (a drink) to` | the one not repeated | the same with no fixed word before the names |
| `mover` | the frame without `S`, `to` flagged `FIRST` or `SECOND` | the name at that place | the name mover alone, inside the frame |
| `dupbit` | the sentence of `ioi`, `to` flagged `WHICH` | `first` or `second`, where the duplicate's earlier mention stood | the duplicate signal through the frame, no name moved |

The duplicate's position varies by the words before it (none, `went`, `went out`) and after it
(`gave`, `gave a drink`). Training holds five of the six shapes; `went S gave to` is read only.

## Panels and conditions

A single function's panel mixes rows by condition. `taught`: every token taught. `key_new`: the
token the read is addressed by is never-taught, the answer taught. `answer_new`: the answer is
never-taught, the key taught. `new`: both. Tokens around them are taught. `copying` reads each
answer's condition from the token heard and the token asked, and its closed loop holds a confound
the panel's forced reading removes: a token the brain cannot say leaves the next moment hearing
another.

Sentence panels come in twins: one pair of names asked twice, each name repeated once, so a read
of a fixed place reads .50 and `both` (twins both right) reads .00.

| Panel | Names |
| --- | --- |
| `ioi` | taught, pairs training shows, training's shapes |
| `ioi_pair` | taught, the pairs never heard together |
| `ioi_shape` | taught, pairs training shows, the shape no lesson reached |
| `ioi_half` | one name of each pair never-taught |
| `ioi_new` | both never-taught: the named task |

**The half panels read no suppression alone.** A pair of one taught and one never-taught name can
be answered by saying the taught one. `which` and `dupbit` read suppression and the duplicate
signal at never-taught names with an answer that needs no route.

## What each row reads

- `answer`: the brain's greedy act is the answer. `says_avoid`: it says the duplicate. `pair`: it
  says one of the two names.
- `first`: the slot head 0 attends most holds the right token as its value or its next. Where
  neighbouring slots hold both names (`suppression`) this reads nothing; `ports` is the reading
  there.
- `ports`, `port_best`: per read port, whether the token it drives most is the right one, and the
  best port.
- `mass`, `mass_avoid`: the attention on slots that hold the right token, and the duplicate.
- `duplicate_read`, at the duplicate's own moment: whether the first choice's value is the
  duplicate (`first_value`) and the attention on its earlier mention (`mass`).
- `terms`: head 0's logits at the duplicate's moment and at the answer, on the slots a suppressing
  read tells apart, split by source: the event against the keys (`key`), the address map's event
  rows, cortex columns and bias (`event`, `cortex`, `bias`), and the same three of the position
  maps (`position_event`, `position_cortex`, `position_bias`).
- `decode` (a life's option): a read-only ridge decoder of which name repeated, fit on taught
  sentences, from the cortex and the trace after the duplicate, before the answer and after it. The
  trace holds the cortex's activity of the moment before, so the two read alike.

## Causality

Each panel is read again on a fork under each intervention, the answer and the read's own reading
kept.

| Intervention | At |
| --- | --- |
| `silence` | the read's amplitude at zero at every moment |
| `silence_answer`, `silence_dup` | at the answer, at the duplicate's moment |
| `erase`, `erase_dup` | the slot each head attends most emptied (key, value, next) before the act, at the answer, at the duplicate's moment |
| `erase_source` | the answer's own slot emptied, and the next of the slot before it: both routes |
| `erase_other` | a control: a slot neither attended nor holding the answer emptied |
| `trace`, `cortex`, `focus` | at the answer: the working trace, the cortex the address reads, the previous read's focus cleared |
| `swap_trace`, `swap_cortex`, `swap_focus`, `swap_slot`, `swap_all` | at the answer, on twin panels: the twin's carrier in the row's place (`swap_slot`: the duplicate's slot); an answer that follows the twin's names the carrier |

## Arms

Each on founders 0 to 2, then on brain seeds (`init_seed`, on the founder's split, lessons and
panels) and last-bit replicas of what moves. Every arm is
`Brain.compose(inputs, actions, modules=(64,), seed=founder)` with the settings below; development.

| Arm | Settings |
| --- | --- |
| `Fd` | the base composition: `episodic=False, recall=True, recall_recency=0.0, learning_surprise=True, recall_boundary=True, recall_anchor=True, recall_slope=0.1, recall_position_rate=0.01, recall_relative=True, teaching="deep"` (the address lesson as a contrast, the library's one law), no apical |
| `Fd2` | `Fd` with `recall_heads=2` |
| `FdT` | `Fd` with the working trace cleared before every lesson and act, as the grammar harness's cut |
| `Fdg`, `Fd2g`, `FdTg` | each with `recall_gain=30.0` and `recall_tokens` on every word |

`Support` (`support` in a life's spec) is the location's ladder and no arm: what the harness gives or
takes from a life, to name a term by what restores the composition.

| Support | What it does |
| --- | --- |
| `no_address_cortex` (address cut) | the address map's cortex columns stay at zero through the life |
| `no_cortex` (cortex cut) | the cortex of the moment before reads zero to every map, at every lesson and act |
| `center_cortex` | the maps read the cortex less its running mean |
| `raw_dup` (oracle) | the read at the duplicate's moment scores the event against the keys alone |
| `gain_at_answer` (oracle) | the gains' drive reaches the motor at answer moments alone |
| `gain_late` | the gains stay at zero through the first thousand lessons; with `gain_late_value` in a life's spec (one number, or one per port) they are set to it then |
| `cortex_unit` | the maps read the cortex of the moment before at unit norm per stream (the emulation of `recall_cortex_norm`) |
| `cortex_scale`, `cortex_rate` (a life's spec) | the maps read that cortex times a number; each lesson's step on every map's cortex columns times a number |
| `gain_start` (a life's spec) | every port's gain at birth, one number or one per port |
| `hold_focus`, `hold_cortex`, `hold_trace` (oracles) | what the duplicate's moment left stands in place up to the answer |

## Controls

[`nets.py`](nets.py): the LSTM and the transformer of the
[grammar chamber's controls](../grammar/controls.py), built by that code at a task's inputs and
actions, their widths the nearest to the `Fd` brain's `parameters()` on that task. They hear the
brain's moments and are taught its answer moments by cross-entropy, teacher-forced; the same scorer
reads them. `online`: one pass, one Adam step per batch at 0.003, equal exposure. `offline`: the
same batches in shuffled epochs at 0.001 until the seconds in updates reach the `Fd` life's on the
same box, equal compute. Causality on a control: a moment blanked at test, and on a transformer the
attention layers' output silenced. They are independently scored controls and never part of a
brain's answer path.

**How a never-taught name reaches the output.** In `lstm` and `transformer` it has no route: its
input weights stay at their random start and its output row is a row of the last linear map that no
lesson raised and every lesson's cross-entropy lowers. The brain without `recall_gain` has none
either. The brain with `recall_gain` has one the two controls lack: a learned scalar per read port
on the token's identity. `tied`, reported apart, gives the transformer the matching route by weight
tying: its logits are the hidden state's product with the same token's input weights. `pointer` adds
one more attention over the moments before, whose read of the heard tokens reaches the logits through
one learned scalar from zero, the brain's route itself, beside the last linear map. Both read .00 at
every never-taught answer: the route exists and goes unused, since the last linear map answers the
taught tokens first. `copy` takes that map from the tokens, so that a token's logit is the log of
that attention's read of it, a pointer network. It is the control whose route is used: a gain reading
above the plain networks at never-taught answers is that route and no more, and a reading against
`copy` compares attention with attention.

## Readings

Development, on one shared 64-vCPU box whose load stood near 80, so seconds are inflated and compare work within a row. A life's split, lessons and panels come from its founder. `Fd`'s lives on the library with `recall_gain` equal those on the library without it, panels and work, on all seven single functions.

**Noise.** A condition holds 96 rows (192 for `duplicate`, 160 for a sentence panel), about .05 at .5 on sampling alone. Last-bit replicas move a function's cell by a mean of .001 and at most .15 (72 lives) and flip no sentence life (12 lives), so the spread that counts is between brains. On six brains `Fd` stays within .07 on every function's cell below; `FdT` spreads by .18 to .22 at a never-taught key on `successor` and `induction`, and `Fd2` by .17 on `duplicate`. Sentence lives are bimodal: a life reads 1.00 or about .50, so a rate over twelve lives is the reading and a mean of three is a screen.

### Single functions

Means over founders 0 to 2. `Read` is the first choice of `first` at both-new. `Silenced` and `Erased` are on taught rows; for a network `Erased` blanks the moment that told the answer. `Control` is `erase_other`.

| Function | Arm | Taught | Key new | Answer new | Both new | Read | Silenced | Erased | Control | Seconds | Parameters |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `previous` | `Fd` | 1.00 | 1.00 | .00 | .00 | 1.00 | .69 | .50 | 1.00 | 63 | 15,326 |
| | `FdT` | 1.00 | 1.00 | .00 | .00 | 1.00 | .11 | .42 | 1.00 | 60 | 15,326 |
| | `Fd2` | 1.00 | 1.00 | .00 | .00 | 1.00 | .52 | .23 | 1.00 | 53 | 24,738 |
| | `Fdg` | 1.00 | 1.00 | .98 | 1.00 | 1.00 | .35 | .48 | 1.00 | 67 | 15,328 |
| | transformer, one pass | 1.00 | 1.00 | .00 | .00 | | .12 | .13 | | 10 | 15,852 |
| | transformer, equal compute | 1.00 | 1.00 | .00 | .00 | | .14 | .14 | | 69 | 15,852 |
| | LSTM, one pass | 1.00 | 1.00 | .00 | .00 | | | .13 | | 4 | 15,564 |
| | `copy`, one pass | 1.00 | 1.00 | 1.00 | 1.00 | | .89 | .16 | | 13 | 15,525 |
| `duplicate` | `Fd` | 1.00 | .92 | | | | .50 | .34 | .86 | 43 | 15,456 |
| | `FdT` | 1.00 | .91 | | | | .50 | .43 | .94 | 41 | 15,456 |
| | `Fd2` | .90 | .94 | | | | .50 | .48 | .88 | 56 | 24,868 |
| | `Fdg` | 1.00 | .98 | | | | .50 | .61 | .74 | 85 | 15,458 |
| | transformer, one pass | .73 | .57 | | | | .52 | .53 | | 11 | 15,914 |
| | transformer, equal compute | 1.00 | .57 | | | | .52 | .52 | | 46 | 15,914 |
| | LSTM, one pass | .50 | .50 | | | | | .50 | | 4 | 15,674 |
| | LSTM, equal compute | 1.00 | .48 | | | | | .50 | | 43 | 15,674 |
| | `copy`, equal compute | 1.00 | .63 | | | | .48 | .51 | | 44 | 15,583 |
| `successor` | `Fd` | 1.00 | .97 | .00 | .00 | 1.00 | .11 | .03 | 1.00 | 40 | 15,326 |
| | `FdT` | 1.00 | .89 | .00 | .00 | 1.00 | .10 | .04 | 1.00 | 40 | 15,326 |
| | `Fd2` | 1.00 | .99 | .00 | .00 | 1.00 | .11 | .04 | 1.00 | 48 | 24,738 |
| | `Fdg` | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | .12 | .01 | 1.00 | 49 | 15,328 |
| | transformer, one pass | 1.00 | .57 | .00 | .00 | | .14 | .09 | | 9 | 15,852 |
| | transformer, equal compute | 1.00 | .65 | .00 | .00 | | .14 | .10 | | 51 | 15,852 |
| | LSTM, equal compute | .89 | .72 | .00 | .00 | | | .01 | | 46 | 15,564 |
| | `copy`, one pass | .98 | .63 | .97 | .56 | | .41 | .06 | | 9 | 15,525 |
| | `copy`, equal compute | 1.00 | .55 | 1.00 | .63 | | .34 | .08 | | 48 | 15,525 |
| `induction` | `Fd` | 1.00 | 1.00 | .00 | .00 | 1.00 | .14 | .12 | 1.00 | 56 | 14,955 |
| | `FdT` | 1.00 | .89 | .00 | .00 | 1.00 | .15 | .11 | 1.00 | 52 | 14,955 |
| | `Fd2` | 1.00 | 1.00 | .00 | .00 | 1.00 | .14 | .14 | 1.00 | 65 | 24,061 |
| | `Fdg` | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | .08 | .09 | 1.00 | 60 | 14,957 |
| | transformer, one pass | .99 | .35 | .00 | .00 | | .13 | .15 | | 13 | 15,822 |
| | transformer, equal compute | 1.00 | .47 | .00 | .00 | | .15 | .11 | | 56 | 15,822 |
| | LSTM, equal compute | .71 | .50 | .00 | .00 | | | .15 | | 55 | 15,348 |
| | `copy`, one pass | .99 | .61 | .93 | .35 | | .30 | .09 | | 11 | 15,497 |
| | `copy`, equal compute | 1.00 | .61 | .99 | .50 | | .36 | .08 | | 57 | 15,497 |
| `copying`, forced | `Fd` | 1.00 | .99 | .00 | .00 | 1.00 | .13 | .10 | 1.00 | 101 | 15,699 |
| | `FdT` | 1.00 | .99 | .00 | .00 | 1.00 | .12 | .06 | 1.00 | 90 | 15,699 |
| | `Fd2` | 1.00 | .98 | .00 | .00 | 1.00 | .12 | .09 | 1.00 | 117 | 25,419 |
| | `Fdg` | 1.00 | 1.00 | 1.00 | .99 | 1.00 | .13 | .00 | 1.00 | 91 | 15,701 |
| | transformer, one pass | 1.00 | .68 | .00 | .00 | | .13 | .79 | | 11 | 15,882 |
| | transformer, equal compute | 1.00 | .71 | .00 | .00 | | .14 | .79 | | 104 | 15,882 |
| | LSTM, equal compute | 1.00 | 1.00 | .00 | .00 | | | .80 | | 96 | 15,780 |
| | `copy`, equal compute | .98 | .69 | .95 | .73 | | .48 | .76 | | 104 | 15,553 |

- **The read finds what no lesson used.** On `previous`, `successor`, `induction` and forced `copying` the first choice holds the never-taught answer on 1.00 of rows in every arm. Without the gain the answer is .00; the failing part is the route, not the selection.
- **The gain carries it on these functions.** `Fdg` says never-taught answers at .96 to 1.00. Silenced they read .00 to .09 and with the attended slot erased .00 to .19, and the control erasure holds, so the read is what says them. On six brains the reading stays .96 to 1.00. The pointer network reads both-new `induction` at .35 and .50 and `successor` at .56 and .63, one pass and equal compute, and `previous` at 1.00.
- **Where the read carries a function and where weights do.** Trained on one fixed order (`successor_fixed`), every arm says that order at 1.00. Silenced the transformer holds .79 and `Fd` falls to .45, so its weights carry the order and the brain's read stays in use. On told orders of their own `Fd` reads .35 and the transformer .14.
- **The trace is a second carrier of `previous`.** `Fd` holds .69 silenced at every moment and `FdT` .11.

**`duplicate`** needs no route. The raw key finds the earlier mention on 1.00 of yes rows, and `Fd` reads .92 at a never-taught query where the transformer reads .57 after equal compute. Silenced, `Fd` says one word on every row (.50). With the trace cleared at the answer it falls to .51 and .63, so the signal reads the read and the trace together.

**Copying in closed loop** reads key new at .04 on `Fd`, a confound and no failure of the read: a token the brain cannot say leaves the next moment hearing another. `Fdg` says it, and reads .99 in closed loop.

**Suppression**, `A B S GO`, with its answers by name and by place:

| Task | Arm | Taught | Both new (per founder) | Port, taught | Port, both new (per founder) | Twin's trace | Twin's focus | Rows taught | Sweeps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `suppression` | `Fd` | 1.00 | .00 .00 .00 | .46 | .17 .88 .00 | .00 | .85 | 542 | .23M |
| | `FdT` | 1.00 | .00 .00 .00 | .93 | 1.00 1.00 .89 | 1.00 | .03 | 2,718 | .84M |
| | `Fd2` | 1.00 | .00 .00 .00 | .73 | .50 1.00 .44 | .01 | .74 | 533 | .23M |
| | `Fdg` | 1.00 | .02 .04 .03 | .51 | .00 .00 .08 | .00 | .24 | 1,218 | .30M |
| | `FdTg` | 1.00 | .00 .50 .11 | .95 | 1.00 .50 1.00 | 1.00 | .01 | 3,653 | .62M |
| | transformer, one pass | 1.00 | .00 .00 .00 | | | | | | |
| | `copy`, one pass | 1.00 | .93 .89 .79 | | | | | | |
| | `copy`, equal compute | 1.00 | .68 .82 .66 | | | | | | |
| `which` | `Fd` | 1.00 | .61 .43 .41 | .47 | .86 .00 .50 | .57 | .53 | 1,322 | .29M |
| | `FdT` | 1.00 | .48 .50 .50 | .25 | .14 .00 .00 | 1.00 | .44 | 1,527 | .33M |
| | transformer, one pass | 1.00 | .78 .91 .86 | | | | | | |
| | transformer, equal compute | 1.00 | .74 .78 .82 | | | | | | |
| | `copy`, one pass | 1.00 | .92 .78 .83 | | | | | | |

Every arm says taught names at 1.00, by different means. With the trace, a twin's trace in a row's place flips the answer (.00) and the read names the answer on .46 of rows: the cortex holds the sentence. Without it, a twin's focus flips the answer (.03) and the read names both-new names at 1.00, 1.00, .89: the read holds it, at five times the rows taught and 3.7 times the sweeps.

### The named task

`ioi`, 4,000 batches, founders 0 to 2 on their own newborns. `copy` is the pointer network below.

| Arm | Heard together | Never together | Unseen position | Both new: answer | Both new: port | Mass at the duplicate | Seconds | Sweeps | Parameters |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `Fd` | .50 .50 1.00 | .47 .50 .94 | .56 .50 1.00 | .00 .00 .00 | .50 .50 1.00 | .01 .00 .71 | 264 | 3.31M | 18,499 |
| `FdT` | .50 .49 .50 | .54 .48 .50 | .53 .48 .49 | .00 .00 .00 | .51 .49 .50 | .01 .03 .02 | 477 | 6.16M | 18,499 |
| `Fd2` | 1.00 .49 .49 | .84 .50 .46 | 1.00 .48 .50 | .00 .00 .00 | .51 .51 .52 | .01 .01 .01 | 370 | 4.02M | 30,109 |
| `Fdg` | .50 .51 .50 | .49 .48 .50 | .50 .50 .50 | .00 .00 .00 | .51 .47 .50 | .01 .01 .01 | 268 | 4.32M | 18,501 |
| transformer, one pass | 1.00 1.00 1.00 | .65 .69 .83 | 1.00 1.00 1.00 | .00 .00 .00 | | | 21 | | 18,420 |
| transformer, equal compute | 1.00 1.00 1.00 | .70 .88 .69 | 1.00 1.00 1.00 | .00 .00 .00 | | | 187 | | 18,420 |
| LSTM, one pass | 1.00 1.00 1.00 | .40 .38 .21 | 1.00 1.00 1.00 | .00 .00 .00 | | | 9 | | 18,612 |
| LSTM, equal compute | 1.00 1.00 1.00 | .09 .16 .34 | 1.00 1.00 1.00 | .00 .00 .00 | | | 183 | | 18,612 |
| `copy`, one pass | 1.00 1.00 1.00 | .85 .75 .78 | 1.00 1.00 1.00 | .69 .93 .63 | | | 41 | | 18,171 |
| `copy`, equal compute | 1.00 1.00 1.00 | .70 .71 .89 | 1.00 1.00 1.00 | .74 .88 .53 | | | 295 | | 18,171 |

Lives on twelve brains (founders 0 to 2, each on its own newborn and three more seeds), with the location's cuts. A passing life reads .95 or above on `ioi`; `Both new` counts lives that say both-new names at .80 or above and gives their readings:

| Life | Passing | Mean, `ioi` | Both new at .80 | Their readings | Rows taught, passing and failing | Sweeps, passing and failing |
| --- | --- | --- | --- | --- | --- | --- |
| `Fd` | 2 of 12 | .58 | 0 |  | 1,164, 32,484 | 1.3M, 4.7M |
| `Fd`, without the anchor | 2 of 12 | .58 | 0 |  | 1,176, 32,109 | 1.3M, 4.5M |
| `Fd`, the cortex less its mean | 1 of 12 | .54 | 0 |  | 1,738, 31,172 | 1.3M, 4.5M |
| `Fd`, address cut | 7 of 12 | .79 | 0 |  | 1,233, 32,025 | 1.3M, 4.9M |
| `Fd`, cortex cut | 7 of 12 | .80 | 0 |  | 3,444, 31,779 | 1.7M, 4.8M |
| `Fd`, address cut, without the slope | 9 of 12 | .91 | 0 |  | 1,461, 25,364 | 1.3M, 4.1M |
| `Fdg` | 0 of 3 | .50 | 0 |  | , 32,329 | , 4.3M |
| `Fdg`, cortex cut | 5 of 12 | .69 | 4 | 1.00 1.00 1.00 1.00 | 7,330, 37,997 | 2.0M, 4.6M |
| `FdTg`, cortex cut | 5 of 12 | .71 | 5 | 1.00 1.00 1.00 1.00 .98 | 2,335, 36,542 | 2.0M, 5.4M |
| `Fdg`, address cut, without the slope | 3 of 12 | .59 | 0 |  | 1,874, 36,648 | 1.3M, 4.4M |
| `Fdg`, cortex cut, without the slope | 4 of 12 | .65 | 3 | 1.00 1.00 1.00 | 9,955, 36,948 | 2.1M, 4.2M |
| `FdTg`, cortex cut, without the slope | 6 of 12 | .72 | 6 | 1.00 1.00 1.00 1.00 1.00 .93 | 3,457, 39,241 | 2.5M, 5.2M |
| `Fdg`, address cut, drive at answers (oracle) | 6 of 12 | .70 | 6 | 1.00 1.00 .99 .97 .94 .86 | 1,640, 37,743 | 1.3M, 4.7M |
| `Fdg`, address cut, drive at answers (oracle), without the slope | 5 of 12 | .61 | 5 | 1.00 1.00 1.00 .86 .82 | 2,106, 38,326 | 1.4M, 5.0M |

- **`Fd` reads the named task's circuit on two lives of twelve**, and the cuts raise that. There one head does what the question asks. This is the read's first choice at each moment, taught and both-new names alike, on three lives that form it:

| Moment | The first name repeats | The second name repeats | Top attention |
| --- | --- | --- | --- |
| `A`, `and`, `B`, and `went`, `out` where heard | `when` | `when` | .47 to .66 |
| the duplicate | the first name's slot | the second name's slot | .63 to .79 |
| `gave` | `and`, or still the first name's slot | `when` | .30 to .45 |
| `a`, `drink` where heard | `and` | `when` | .55 to .67 |
| `to` | `and`, whose next is the second name | `when`, whose next is the first name | .65 to .79 |

  The raw key is the duplicate detector: it finds the earlier mention. The next word's read is the suppression: it steps by the relative map from the earlier mention to the slot whose next holds the other name, and stays there across the words between. The next port is the name mover. It holds with the trace silent and the cortex read as zero, so the read's own place is what crosses the moments, and no lesson falls on any moment but `to`. On the seven lives of `Fd` that form it under the address cut:

| At test | Taught: answer | Both new: read's port |
| --- | --- | --- |
| The read silenced at every moment (its attention stays) | .09 to .18 | 1.00 |
| Silenced at the answer | .27 to .64 | 1.00 |
| The attended slot erased at the answer | .08 to .21 | .00 to .10 |
| The answer's slot lesioned on both routes | .14 to .30 | .00 |
| A slot neither attended nor holding the answer erased | 1.00 | 1.00 |
| The attended slot erased at the duplicate's moment | .40 to .50 | .40 to .50 |
| Silenced at the duplicate's moment | .99 to 1.00 | 1.00 |
| The focus cleared at the answer | .52 to .67 | .38 to .50 |
| The cortex cleared at the answer | .56 to .86 | .50 to 1.00 |
| The trace cleared at the answer | .72 to 1.00 | 1.00 |
| A twin's focus in the row's place | .00 to .08 | .00 |
| A twin's cortex | 1.00 | 1.00 |
| A twin's trace | .89 to 1.00 | 1.00 |
| A twin's duplicate slot | 1.00 | 1.00 |

- **The named task, whole, reads at names no lesson used** with no oracle on 4 of 12 lives of `Fdg` under the cortex cut, 5 of 12 of `FdTg` under it and 6 of 12 of `FdTg` under it without the slope, at .93 to 1.00, with twins both right on .96 to 1.00 under the cortex cut. At both-new names those lives read .00 silenced and .00 with the attended slot erased, .50 with the slot erased at the duplicate's moment and .00 with a twin's focus in the row's place, and the control erasure holds their reading. A life that misses reads about .50 on taught names and .00 on new ones. Pairs of one taught and one new name read 1.00 when the new one repeats and .04 to .99 when the taught one does: there the association votes for the taught name.
- **The pointer network reads it at .75 and .71** at one pass and at equal compute, .53 to .93 per founder, with no life at a coin flip on taught names. It has what the brain has not: credit across moments.
- **The plain networks say taught sentences at 1.00 and never-taught names at .00.** On pairs never heard together, the brain's lives that form the circuit read .94 to 1.00 against .65 to .88 for the transformer and .09 to .40 for the LSTM.
- **The parts alone.** The name mover, told by a cue, reads 1.00 on all three founders of `Fd` and names both-new names at 1.00; the failure is the selection, not the moving.

**How it compares.** A passing life of the brain takes about 1.3M sweeps and 190 seconds on this box, a failing one 4.7M and 310, against 21 seconds for the transformer's pass and 187 at equal compute, at the same parameters. On taught sentences the plain networks read 1.00 on every life where the brain's base composition reads 1.00 on 2 of 12. At names no lesson used they say nothing, since they have no route, so the comparison there is the pointer network:

| Both new | `Fdg` | `copy`, one pass | `copy`, equal compute |
| --- | --- | --- | --- |
| `previous` | 1.00 | 1.00 | 1.00 |
| `successor` | 1.00 | .56 | .63 |
| `induction` | 1.00 | .35 | .50 |
| `copying`, forced | .99 | .74 | .73 |
| `suppression` | .03 | .87 | .72 |
| `which` | .55 | .84 | .78 |
| `ioi` | .00 on every life; under the cortex cut 1.00 on 4 of 12, .00 on 8 | .75 | .71 |

The brain's read is the better finder of a token by content: raw keys hold at a token no lesson used, where a learned embedding does not. The pointer network is the better composer: it reads the duplicate at both-new names on every founder, where the brain forms the circuit on some lives and on those says it at 1.00. That reading is the whole of the comparison here: a next-word network trained by gradients across the sentence, against a brain taught by local rules at the answer alone.

### Where it fails

Each failure, by the term that fails.

1. **The read's selection, by the maps' cortex columns.** At `ioi`, lessons name the slots whose next holds the answer. In this frame those are `when` and `and`, the same keys in every sentence. The address map steps on `z = [x, c, 1]`, the cortex holds about twenty times the event's energy (`|c|^2` 18 to 26 against 1 on failing lives), and it stays nearly the same from moment to moment (cosine .86 to .90 between the duplicate's moment and the answer's). So the cortex columns take most of the step, and what they learn at `to` applies at every moment. At the duplicate's moment on a failing life the raw key gives the earlier mention +4.0 and the slot before it gets +4.4 from the address's cortex columns and +3.0 from the position maps', mostly their cortex columns. The first choice leaves the earlier mention (mass .005), and neither the read nor the cortex holds which name repeated (decoder .53). At `to` the two fixed slots read alike, so the answer is a coin flip between the names. Passing lives hold those columns near +1.1.
   - The frame without the fixed words (`bare`) reads 1.00 on `Fd` and `FdT` on all three founders, with the read naming both-new names at 1.00.
   - Holding the address's cortex columns at zero passes 7 of 12. On the five that miss, the position maps' cortex columns take the pull (+4.5 to +5.8 against +1.2 to +1.9). The cortex read as zero by every map also passes 7 of 12, and there the bias rows take it. Without the slope map the address cut passes 9 of 12.
   - Finding the duplicate alone is not enough: an oracle raw read at its moment leaves `ioi` at .51 and .49 where the life fails, since the answer's read keeps its pull. Without the next port the pull goes (mass .97) and so does the name mover (.57).
   - The cortex less a running mean passes 1 of 12, so the shared mean alone does not name it. Without the anchor the rate does not move (2 of 12).
2. **The duplicate signal through the frame, as a word.** `dupbit` reads .50 on `Fd`, where the read finds the mention (.68); the bit does not reach `to` for a word to say it. The read's focus carries it where the answer is a slot, and nothing reliably carries it where the answer is a word: under the address cut `dupbit` reads .50, 1.00, .80 and both-new .50, .54, .80. `which` at both-new names reads .41 to .61 on `Fd` where a transformer reads .74 to .91, and `recall_when`, which tells the cortex where a read landed, does not lift it (.61 .43 .50).
3. **The gain's vote.** Where the read selects, the gain's drive decides at never-taught names, since the association reads near zero there. The vote fails in three ways:
   - In `suppression`, every slot near the names holds a name on one port and the other on the other, and the duplicate holds two slots' worth: its drive exceeds the answer's (+1.99 against +1.53), and the brain says it on .96 to .98 of rows. `ioi` avoids this: the slot before a name holds a frame word as its value.
   - In `bare` the slot the read names holds the answer on its next and, where the duplicate came first, the duplicate as its value: answer and duplicate tie (+2.28 against +2.31) and the brain reads .42 to .51, saying the duplicate on .81 to .83 of those rows.
   - At every moment the gain drives the read's token into the motor, and that reaches the cortex the position maps read. On `ioi` with the address cut, the one life of three that forms the circuit then points the read at the duplicate on never-taught names (port .01), and the read returns to 1.00 when it is silent at every moment. With the gains' drive at answer moments alone, an oracle, it reads 1.00, 6 of 12 lives pass and all six say both-new names at .86 to 1.00. The cortex read as zero by the maps gives the same without an oracle.
4. **The life's first lessons.** The gain lowers the rate at which lives form the circuit, and lives that miss are not a little wrong: their association reads +8 to +14 on a taught name and -9 to -16 on the new ones, against about 0 on lives that form it. Holding the gains at zero for the first thousand lessons leaves them near zero for good on lives that form the read, since the surprise rule then teaches no rows.
5. **Two heads.** `Fd2` passes `ioi` on one founder of three by another route: a twin's duplicate slot at `to` flips the answer (.02), and both-new selection stays at chance (.51). The duplicate's slot is read at the answer and the cortex compares by token, which does not reach names no lesson used. Under the address cut with the gain, on the life that selects both-new names (port 1.00), the second head's port names the duplicate on about half the rows, the vote ties (+1.83 against +1.80) and the answer reads .57.

**A cost of the cuts.** A read on a read goes through the cortex. `twohop`'s second hop at taught tokens reads 1.00, .99, 1.00 on `Fd`, .40, .42, .50 under the address cut and .27, .42, .44 under the cortex cut, with rows taught up from about 2,800 to about 14,300; the first hop holds at 1.00. So the columns that lose the duplicate are the path of a read on a read, and neither cut can be delivered as it stands. On the single functions the address cut costs nothing above noise: `previous` 1.00, `successor` 1.00 and .99, `induction` 1.00, forced `copying` 1.00, `duplicate` .99 and .91.

**A read on a read at a never-taught key is lost on every arm.** `Fd`'s second hop reads .05 to .12 with the read's first choice at .00 to .02, and `Fdg` .01 to .03: the address reaches a slot through columns taught on taught tokens only. Where the key is taught and the answer new, `Fdg` reads .67 to .79 with the read at 1.00.

## The cortex at the event's weight and a value path released late

Two library opt-ins come from the first failure term and the gain's. `recall_cortex_norm=True` gives
every map of the read the cortex of the moment before at unit norm, so the fixed slots a lesson at `to`
names no longer take the step at every moment. `recall_gain_after=n` with `recall_gain_next=g` holds
the learned gains at zero through the first `n` lessons and starts the next port's gain at `g` then,
the recall port's at zero ([the memory guide](../../docs/memory.md#the-value-path-released-once-the-read-has-formed-opt-in)).
On this chamber the harness's emulation of the second (`gain_late` with `gain_late_value`) and the
option read the same panels and work to the bit (`ioi` and `twohop`, two founders each).

The readings below are declared: a protocol written before any life, twelve fresh founders, each on
its own newborn, one machine (a 32-vCPU box), `causal` readings off. A life passes the gate where
never-taught names (`ioi_new`) are said at .80 or above and the taught second hop (`twohop`,
`hop2_taught`) reads .90 or above on the same founder.

| Reading | Arm | Taught `ioi` at .95 | The read's port at never-taught names, .80 | Never-taught names said, .80 | Taught second hop, .90 | Lives passing |
| --- | --- | --- | --- | --- | --- | --- |
| First (founders 2691001 to 2691012) | `Fd` | 2 | 2 | 0 | 12 | 0 |
| | `Fd`, cortex at unit norm | 7 | 7 | 0 | 12 | 0 |
| | `Fdg`, cortex at unit norm, both gains released at 30 after 1,000 lessons | 6 | 6 | 6 | 0 (.00 to .05) | 0 |
| Second (founders 2691013 to 2691024) | `Fd`, cortex at unit norm | 7 | 7 | 0 | 12 | 0 |
| | `Fdg`, cortex at unit norm, `recall_gain_after=1000`, `recall_gain_next=10` | 7 | 7 | 7 (.97 to 1.00) | 12 (.99 to 1.00) | 7 |

The bar of 8 of 12 is missed by one life. What each reading shows:

- **The cortex at unit norm moves the circuit from 2 to 7 lives of 12** and keeps the taught second hop
  on every life (.98 to 1.00). On the lives that form it the read's port names never-taught names at
  1.00; without a value path nothing says them.
- **A gain on the recall port breaks the second hop.** In `twohop` the read lands on the slot of the
  query, whose value is the query itself: a recall port's gain says it, and with both gains at 30 the
  second hop reads .00 to .05 on all twelve lives and the first .52 to .86.
- **The gain from birth stops the read from forming** (development, founders 0 to 5: the next port's
  gain at 10 from birth forms the circuit on 1 or 2 lives of 6, and the second hop reads .00 to .45). A
  gain that answers early takes the rows the read's lesson needs. Released after 1,000 lessons it
  forms on 6 of 6 there, and on 7 of 12 declared.
- **The five lives that miss fail at the read's selection.** On three of them, at the duplicate's moment
  the position maps give the slot before the earlier mention +1.0 to +1.7 against the content's lead of
  1.4 to 1.7 for the mention itself, so the read holds the earlier mention at .27 to .47 where passing
  lives hold .53 to .86 (their content's lead 2.1 to 3.3). On the other two the read at `to` lands on the
  names' own slots, the content (+2.5 to +3.0) and the position maps (+3.8 to +4.0) both favouring them
  over the slot before. A passing life settles an act in 32 to 43 sweeps, a failing one in 39 to 67.

On the library of this release the chamber reads the earlier rows above as they stand (development,
founders 0 to 2): `ioi` on `Fd` .50, .51, 1.00 on taught sentences; the second hop 1.00, 1.00, .99; the
single functions' cells within .05 of the table above on `Fd` and `Fdg` (copying read in closed
loop).

## What this does and does not establish

The chamber reads one composition and its options on one finite language of twenty words. A sentence said right at names no lesson used shows that the life composed the parts on this frame, on its development founders; it shows nothing about open vocabularies, other frames or fresh founders, and the cuts and oracles above are diagnostics that name a term. None is a candidate as it stands: the cortex cut costs the second hop, and the oracle knows which moment is an answer. The rates are twelve lives on one machine and carry the width of twelve bimodal lives. What decides a life's side is not established: replicas keep it and brain seeds change it, and the first lessons differ by the end (rows taught, the cortex's energy) without naming a cause. With tokens declared, `FdTg` reads `duplicate` and `which` at .50 where `Fdg` reads .95 to 1.00 and `FdT` without tokens reads 1.00; that is read and not located. The controls are next-word learners at the brain's parameter count: a control that says a panel right shows that the information suffices at that size, and `tied`, `pointer` and `copy` are additions of this chamber, reported apart, that compare routes and restrict nothing of the brain's. Seconds on the shared box compare work within a row only.

## Run

From the repository root, with one BLAS thread per process and `torch` for the controls:

```sh
PYTHONPATH=src python benchmarks/heads/heads.py --task ioi --arms Fd FdT Fd2 --founders 0 1 2 \
    --out results/ioi.jsonl --workers 9
PYTHONPATH=src:benchmarks/heads python benchmarks/heads/nets.py --task ioi --models lstm transformer \
    --modes online --founders 0 1 2 --out results/nets.jsonl --workers 3
python benchmarks/heads/heads.py --table results/ioi.jsonl
```
