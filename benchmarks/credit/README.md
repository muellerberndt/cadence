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
tuple. `--erase` adds held-out readings at the last budget with one memory's stream emptied
just before the answer (`working_memory`: the trace; `recall`: the slots). The training log,
per window of 64 episodes: the share right as acted (`reward`) or as the brain would have
answered before the lesson (`lesson`, read on a fork), and under `reward` the share of answers
sampled away from the complete greedy choice with the share right among greedy and among
explored answers.

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
    --arms founder recall_recency0 --delays 2 4 8 --erase working_memory recall \
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

**Preservation** of the three reward settings (`preservation.py`, seeds as above, against each
chamber without the setting): odour nursery `live`, lives passing every check 10 of 12 against
11 (the hold 11), `step` 8 against 7; key-door rule A acquired 16 of 16 against 8; bandit with
reversal 30 of 30 blocks against 27; contextual bandit 6 of 6 seeds against 3; delayed response
6 of 6 against 6. No fall of two or more. In the nursery two lives at the 300-trial exposure
(seeds 2 and 5) reverse late with `slow_episodic` (lags 355 and 482 trials, against 97 and 34
without it); the hold avoids those two and reverses late on one life at 1,000.
