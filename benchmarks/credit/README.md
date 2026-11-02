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

A life at delay 0 takes about a minute on one laptop core, a lesson life at delay 8 a few.
