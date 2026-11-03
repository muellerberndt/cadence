# Two-cue chamber: credit at an identical answer

This chamber asks whether a composed brain learns a choice that depends on a cue it saw
earlier, when the moments in between and the answer's own observation are the same for both
cues. It is the instrument of [issue 208](https://github.com/muellerberndt/cadence/issues/208)
(reward credit at the answer: the terminal reward alone) and
[issue 209](https://github.com/muellerberndt/cadence/issues/209) (credit from the answer to the
earlier states that carried the cue: a teacher lesson at the answer alone). It measures
settings of the released library against the composed founder; it establishes no default.

## The task

Twelve one-hot inputs:

| Inputs | Tokens |
| --- | --- |
| 0, 1 | the cue pair `C0`, `C1`, the one trained here |
| 2, 3 | a second pair `C2`, `C3`, never shown by these conditions |
| 4 to 9 | six distractors `D0` to `D5` |
| 10, 11 | the suffix `S1`, `S2` |

Two actions. An episode at delay `k` is `k + 3` events: one cue of the pair, then `k`
distractors, then `S1`, then `S2`. The right action at `S2` depends on the cue alone: within a
pair the two cues take the two actions, the assignment counterbalanced across founders (bit 0
of the founder number flips the first pair, bit 1 the second). No event after the cue repeats
it, and the suffix is identical for both cues.

The training schedule shows both cues of the pair in a random order in every block of two
episodes; each episode draws its distractors independently and uniformly from the tuples
permitted in training. At delays 2 and more one distractor tuple in four, chosen by a
founder-salted digest of the tuple, is never trained, and the held-out panel reads those
tuples. At delays 0 and 1 no tuple can be held out without holding out a token, so there the
held-out panel is the training distribution itself (the row says `nothing_held_out`). A panel
holds every tuple of its kind where the kind has at most 256 of them, else 64 distinct tuples
drawn from the panel seed, and each tuple is read with both cues: 2 episodes at delay 0, 12 at
delay 1, about 54 and 18 (train and held-out) at delay 2, 128 each at delays 4 and 8.

## Two information conditions

| Condition | What arrives, and when |
| --- | --- |
| `reward` | One continuing stream through `step`. Every act is the library's sampled act; every moment within an episode brings a zero reward for the action before it. The answer's outcome, +1 right and -1 wrong, arrives with the next episode's first event together with `done`, so the library clears the carried context between attempts. Every outcome is learned from. |
| `lesson` | A fresh stream per episode (`reset`). Every event before the answer is a greedy act. At `S2` one `step(x, teacher=[label])`, whose own act is the library's sampled act. Nothing else teaches; the lessons follow the composed rule (every row is taught). |

With two actions the label and the signed terminal reward carry the same information.

## The brain and the arms

Every arm is `Brain.compose(12, 2, seed=founder, arousal=True)` at the composed defaults with
the arm's settings beside it:

| Arm | Settings |
| --- | --- |
| `founder` | none |
| `slow` | `actor_eta=0.1, actor_eta_bias=0.01` (the actor step of [issue 198](https://github.com/muellerberndt/cadence/issues/198)) |
| `slow_episodic` | `slow` with `memory_choice=True` |
| `slow_episodic_extinction` | `slow_episodic` with `memory_extinction=0.5` |
| `slow_episodic_hold` | `slow_episodic` with `memory_context_hold="unchanged"` |
| `recall` | `recall=True` at its founder genes |
| `recall_recency0` | `recall=True, recall_recency=0.0` |
| `recall_recency0_boundary` | `recall_recency0` with `recall_boundary=True` |

## Readings

At 256, 1,024 and 2,048 episodes the train and held-out panels are read on a fork of the life
(`copy.deepcopy`) that never changes it: each panel episode on a fresh stream of the fork
(`reset`, the records kept), every event a greedy act, the answer the greedy act at `S2`. A
reading at budget `n` of the reward condition has learned from `n - 1` outcomes. Per panel:
accuracy, the share answering action 1 per cue, the readout's probability of action 1 per cue,
the episodic records' preference for action 1 per cue (arms with `memory_choice`), and the
relative separation of the two cues' association activity after every event, matched by
tuple. `--erase` adds held-out readings at the last budget with carriers emptied just before
the answer: a memory's stream (`working_memory`: the trace; `recall`: the slots and the cortex
of the moment before), several joined by `+` (`working_memory+recall`: both), and `activity`,
the stream's settled state, so the answer settles from rest. `--transplant` adds a held-out
reading per name with a donor's carriers: for every panel episode a fresh stream of the fork
lives the episode of the other cue with the same distractors up to the answer, the named
carriers' stream state is kept, a fresh stream lives the episode itself up to the answer, the
donor's state replaces its own, and the greedy answer at `S2` is read as following the
donor's cue or its own (the two cues take the two actions, so the shares sum to one). The
training log, per window of 64 episodes: the share right as acted (`reward`) or as the brain
would have answered before the lesson (`lesson`, read on a fork), and under `reward` the share
of answers sampled away from the complete greedy choice with the share right among greedy and
among explored answers.

**Gate.** A setting learns a delay when held-out accuracy at 2,048 episodes is at least .90 on
at least 5 of the 6 development founders (0 to 5). Founders 500 to 511 are kept for a
confirmation declared before it runs. A refused act or lesson ends the life and counts as a
failure.

## Preservation

`preservation.py` reads the reward chambers a setting must keep, with the setting's options
added to every `Brain.compose` call of each chamber's own code and the chamber otherwise
unchanged: the [odour nursery](../reversal/README.md) (arms `live` and `step`), the
[key-door nursery](../keydoor/README.md) (arm `live`), a two-armed bandit with reversal
through `step` (`Brain.compose(1, 2, seed)`, the paying arm +1 and the other -1, the paying arm
moving every 200 of 1,000 trials), and the continuing contextual bandit and delayed response
of `tests/test_context_defaults.py`. A setting breaks a chamber when the lives, blocks or seeds
passing its reading fall by two or more on the same seeds.

## Run

From the repository root, with one BLAS thread per process:

```sh
PYTHONPATH=src python benchmarks/credit/two_cue.py --condition reward \
    --arms founder slow_episodic --delays 0 1 2 --out DIR/reward.jsonl --workers 3
PYTHONPATH=src python benchmarks/credit/two_cue.py --condition lesson \
    --arms founder recall_recency0 --delays 2 4 8 \
    --erase working_memory recall working_memory+recall \
    --transplant working_memory recall working_memory+recall \
    --out DIR/lesson.jsonl --workers 3
python benchmarks/credit/two_cue.py --summary DIR/reward.jsonl
PYTHONPATH=src python benchmarks/credit/preservation.py --chamber nursery --arm live \
    --exposure 300 --settings '{"memory_choice": true}' --out DIR/nursery.jsonl
python -m pytest -q benchmarks/credit
```

A reward life at delay 0 takes 10 to 60 seconds on one core, a lesson life at delay 8 25 to
80 seconds, depending on the host's load.

## Readings on 0.93

Development founders 0 to 5, then founders 500 to 511 for the settings that passed, each
confirmation declared before it ran. Founders at held-out .90 at 2,048 episodes, the mean
held-out accuracy in brackets.

**Terminal reward through `step`** ([issue 208](https://github.com/muellerberndt/cadence/issues/208)):

| arm | delay 0 | delay 1 | delay 2 | delay 0, founders 500 to 511 |
| --- | --- | --- | --- | --- |
| `founder` | 0 of 6 (.50) | 0 of 6 (.54) | 0 of 6 (.50) | 2 of 12 (.58) |
| `slow_episodic` | 6 of 6 (1.00) | 4 of 6 (.92) | 0 of 6 (.51) | 11 of 12 (.96) |
| `slow_episodic_extinction` | 6 of 6 (1.00) | 4 of 6 (.92) | 0 of 6 (.51) | 11 of 12 (.96) |
| `slow_episodic_hold` | 6 of 6 (1.00) | 4 of 6 (.92) | 0 of 6 (.56) | 11 of 12 (.96) |

The composed founder's associative record is keyed on the observation, which both cues share
at `S2`, and every outcome writes it: its greedy answers are right on about a quarter of the
first episodes and its readout ends with one policy for both cues. With the slower step and
the episodic memory the context records split the cues at the shared answer and the readout
follows. `memory_extinction` never acts here (its lives are byte-identical to
`slow_episodic`'s: zero outcomes fall where nothing was paid directly), and
`memory_context_hold="unchanged"` acts only on two equal distractors in a row, so at delays 0
and 1 its lives are `slow_episodic`'s too. From one distractor on, the context key keeps half
of the cue per moment and the readout latches on one action for both cues.

**Where the founder's updates go** (`founder` at delay 0, founders 0 to 5). A read-only
recorder beside the library splits every update of the actor into parts and rebuilds the
library's own change of every synapse and bias from them to within 1.2e-16; the recorded lives
are the chamber's lives exactly. The reward step moves a synapse by `actor_eta` times the
dopamine times its eligibility, which sums over the attempt's sampled acts, decayed by `gamma
lam` per act, the presynaptic activity times the postsynaptic nudged contrast (the forward
term) and the presynaptic nudged contrast times the postsynaptic activity (the feedback term);
a bias moves on its own neuron's contrast, as a synapse from an input that is always one. The
composition has no sensory-to-motor projection: the readout is the association-to-motor
projection, tied to its reverse, and the motor bias. Its outcome updates (12,282), read on the
preference at the answer (motor 1 less motor 0, in drive units) along the association activity
both cues share there and along half their difference:

| part of the update | share of the summed absolute shared change |
| --- | --- |
| forward term, the answer's act | .51 |
| forward term, `S1`'s act and the cue's (through the eligibility) | .21 and .08 |
| motor bias, the three acts | .18 |
| feedback term, the three acts | .02 |

Summed over the updates the shared change is 40 times the cue change (173 against 4.3 drive
units; 3.6 times along unit directions of the association's activity). A forward part's shared
over cue change equals its presynaptic activity's projection on the shared direction over its
projection on the cue direction (median 24 to 301 per founder at the answer's act), whatever
the dopamine and the contrast, which multiply both alike, and a bias part has no cue
component: the term that makes the shared component is the presynaptic activity, the
association activity both cues share at the answer (with the bias's constant input). Outcomes
of either sign carry equal halves of it (86.4 and 86.3) whose nets cancel (-4.4 and +3.8), so
the shared preference wanders (net over root of squares -.39 to +.26 per founder) with steps
40 times the cue's, while the cue part points the right way on 6 of 6 founders (+.05 to +.74
drive units net) and stays small. The nudged contrast sets the level instead: its common mode
over the two motor neurons is not zero, since the activation has two slopes, positive for a
greedy act and negative for an explored one, and the founder's greedy answers are mostly wrong
under the associative record's cue-blind vote (right .25 to .42, explored .60 to .72), so
outcomes of both signs push the common motor drive down (-1.0 to -2.1 drive units on 6 of 6).
The motor potentials fall from about -.3 to between -2.5 and -3.5 on five founders, onto the
activation's flat negative branch, where the cue part barely moves the policy (p(1) .50 to .54
for both cues at the end); founder 2's readout latched on one action instead (.995 for both).
Meanwhile the shared activity at the answer grows (norm .17 to .31, then .82 to 1.92) while
the difference stays (.03 to .05, then .05 to .08): the separation falls on 6 of 6 (.22 to .49,
then .05 to .20), and later updates are more shared still. The readout moves by about twice its
initial size. The projections into the association change the cues' drive at the answer 10 to
40 times more along the activity both cues share than along their difference, the sensory one
not at all along the difference. Updates inside an episode (zero reward) carry 2% of the
summed shared change.

**Exact-gradient audit at delay 0.** An actor-critic with exact gradients on a tiny model of
the same structure (twelve inputs; an association layer of 64 that reads its own previous
activity through a prefrontal layer at the working trace's amplitude, cleared when an attempt
begins; two motor units with the brain's activation, read at temperature .2; REINFORCE with a
critic's baseline, one update per episode; a control, never installed in a brain), trained on
this chamber's delay-0 schedule with the terminal reward, learns the answer on 6 of 6
development founders from a fresh start and on 6 of 6 from the composed founder's own weights
(rate .01, sized on founder 1000; greedy answers right for good from 448 to 1,344 and from 192
to 576 episodes, sampled answers right on .95 to 1.00 at the end). The same exact rule on the
composed founder's association states frozen at birth (cosine .89 to .98 between the cues at
the answer) reads 0 of 6 at every rate from .01 to .3 while the readout's bias steps at the
synapses' rate (the bias walks to one action for both cues and the policy latches there), and
4 of 6 through the motor activation and 5 of 6 with the softmax on the drive at the founder's
bias ratio (.05), sampled answers right on only .48 to .77. The structure learns the cue from
the terminal reward under exact credit; by the issue's rule the brain's local update is the
suspect, and its frozen states at birth carry the answer only by a thin margin.

**A lesson at the answer alone** ([issue 209](https://github.com/muellerberndt/cadence/issues/209)):

| arm | 2 distractors | 4 | 8 | 4 and 8, founders 500 to 511 |
| --- | --- | --- | --- | --- |
| `founder` | 0 of 6 (.66) | 0 of 6 (.50) | 0 of 6 (.50) | 0 and 0 of 12 |
| `recall` | 6 of 6 (1.00) | 6 of 6 (1.00) | 0 of 6 (.53) | |
| `recall_recency0` | 6 of 6 (1.00) | 6 of 6 (1.00) | 5 of 6 (.98) | 12 and 12 of 12 |
| `recall_recency0_boundary` | 4 of 6 (.93) | 6 of 6 (.99) | 6 of 6 (1.00) | 12 and 12 of 12 |

The working trace keeps .55 to .64 of the two cues' separation per later event: from about
1.2 after the cue it falls to .06 to .11 before the answer at 4 distractors and .01 to .02 at
8, and the answer's lesson has nothing to read. The recall memory re-presents the cue at every
moment, and every-row lessons at the answer then reshape the carried state (separation before
the answer .5 to 1.4 in the passing lives, median 1.27). Its founder age charge of .25 per
moment leaves the cue's slot too faint at 8 distractors; with no charge the slot reaches the
answer. Run on the binding chambers' continuing life (token binding without resets, three
trials in twelve slots), `recall_recency=0.0` keeps that life on 6 of 6 seeds, as the recall
memory at its founders does; the same memory without its position term (`recall_position=0`)
loses it at recency .05 and at 0 (0 of 6), the loss recorded for recency .05 before the
position term existed.

**Erasure and transplant** (`recall_recency0`, the development lives of the table above read
again with `--erase` and `--transplant`; every intact reading repeats exactly). Held-out
accuracy at 2,048 episodes with carriers emptied just before the answer, and the share of
answers that follow a donor's carriers, over founders 0 to 5:

| reading | 4 distractors | 8 distractors |
| --- | --- | --- |
| intact | 1.00 on 6 | 1.00 on 5, .85 |
| trace emptied | 1.00 on 6 | 1.00 on 5, .89 |
| slots emptied | .97 to 1.00 | .90 to 1.00 on 5, .56 |
| trace and slots emptied | .50 on 3; .98 to 1.00 on 3 | .50 on 3; .90 to 1.00 on 3 |
| trace, slots and the settled activity emptied | .50 on 6 | .50 on 6 |
| the donor's trace and slots | followed 1.00 on 6 | followed 1.00 on 5, .85 |
| the donor's trace alone | followed .00 to .88 | followed .00 to .23 |
| the donor's slots alone | followed .06 to .97 | followed .52 to 1.00 |
| the donor's settled activity alone | followed .00 on 6 | followed .00 to .15 |

The donor's trace and slots decide the answer against the recipient's own settled state on
every founder; neither alone does on most. With both emptied the answer falls to chance on
founders 0, 1 and 3 (.50: both cues of a tuple then answer alike) and stays on founders 2, 4
and 5: the answer settles from the settled state of the moment before, which the cue shaped,
and on those three that state alone keeps the answer in the cue's basin. With the settled
activity cleared as well the answer settles from rest and every founder answers at chance.
Read with the task-state criteria (held-out at least .90, the two memories emptied at most .60,
the donor followed on at least .80), 3 of 6 founders meet all three at 4 distractors and 2 of 6
at 8, where issue 209 asks 5 of 6: the trace, the slots and the settled activity of the moment
before carry the cue to the answer together.

**A chain's intermediate actions** (a chain whose earlier decisions own their actions, the
preservation reading of issue 209). The [prerequisite chain](../chain/README.md) taught by
lessons along the teacher's path: `Brain.compose(16, 6, seed=founder, arousal=True)`, founders
0 to 5 (the world's seed is the founder), 512 episodes, each a fresh stream from stage 0; at
every stage the greedy answer is read on a fork, then the stage's own action is taught
(`lesson`) and lived by a greedy act. Over the last 64 episodes the intermediate stages and the
last are right on 1.00 on every founder, with the stage observed and hidden, with and without
`recall=True, recall_recency=0.0` (over the first 64 episodes .94 without and .96 with). In the
chain chamber itself, where the arena founder learns the chain from reward through `live` in
one continuing stream (`gene`, observed mode, seeds 0 to 5, 20,000 moments), the same two
settings added to its `Brain.compose` call lower the intermediate stages' greedy answers read
on reset copies at the last probe: .50 to .44 under pay at the end, .78 to .61 under shaped pay
and .39 to .28 under the backward lessons, more than the .05 this reading allows. The last
stage stays right on every life, and the greedy policy run from stage zero completes the chain
as often (66 against 57, 214 against 215 and 22 against 24 per 1,000 moments). The recall
memory at its founder recency moves the same readings (.28, .72 and .39; under pay at the end
it also completes less often, 24 against 57): in a stream whose slots hold its past a stage's
answer leans on them, and on a fresh copy, with none, several lives answer an intermediate
stage with the final stage's action.

**Preservation** of the three reward settings (`preservation.py`, seeds as above, against each
chamber without the setting): odour nursery `live`, lives passing every check 10 of 12 against
11 (the hold 11), `step` 8 against 7; key-door rule A acquired 16 of 16 against 8; bandit with
reversal 30 of 30 blocks against 27; contextual bandit 6 of 6 seeds against 3; delayed response
6 of 6 against 6. No fall of two or more. In the nursery two lives at the 300-trial exposure
(seeds 2 and 5) reverse late with `slow_episodic` (lags 355 and 482 trials, against 97 and 34
without it); the hold avoids those two and reverses late on one life at 1,000.
