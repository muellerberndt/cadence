# Sentences said from a message the brain holds: declared protocol 2 (cadence#286)

Declared on 2026-11-10, committed to the lane branch before the first declared life. The founders,
arms, readings and gates below are fixed; a failed gate is reported as failed. It follows
[the first protocol](MEANING_PROTOCOL.md), whose message an instrument computed and wrote onto
neurons before every settle.

## The question

Said from the instrument's message alone, with the recall read barred from the meaning, clauses
read .99, .95 and .95 (trained, held-out combinations, roles) at 256 association neurons. Three
questions follow for a message the brain writes itself:

1. Does the brain say the sentence from its own message at the same bars?
2. Does a read of one block of the message bring those bars to the default composition's width
   (64 association neurons), where the whole message at once fails new combinations?
3. What does the message cost where the recall read is free? The instrument's message cost paths
   .10 to .40 and three hops .24 to .27.

## The opt-in

`Brain.compose(message=units)`, off by default (`None`). Without it a brain holds no message and
is the founder's to the bit.

**The local rule.** After each moment the brain hears, the moment's fixed sparse code (of `units`
units, the `message_winners` that read most of the moment's active ON and OFF channels) latches
onto block `t mod message_phases` of a region of the brain's own neurons, `t` the moment's place in
the episode, and stays on until the episode ends. The association reads the region through plastic
synapses the patch rule teaches, with one learned gain per block (`carried_gain`); nothing outside
the brain writes onto a neuron, and the write holds no learned parameter.

| Setting | Founder | What it sets |
| --- | --- | --- |
| `message` | `None`, off | units of a block |
| `message_phases` | 4 | blocks |
| `message_winners` | 5 | units set to one per moment |
| `message_fan_in` | 16 | channels each unit reads |
| `message_amplitude` | 1.5 | drive of a unit that is on |
| `message_read` | `False` | a read of one block at a time, below |

**A read of one block (`message_read=True`).** A port of `units` neurons holds
`3 * sum_k p_k block_k`, with `p = softmax((x W_x + c W_c + w) . block_k / 0.25 + turn_k)`: `x` the
moment as heard, `c` the association's settled activity of the moment before, `turn = x T_x + c T_c
+ t` a term per block. The maps start at zero. After a lesson each block is scored against the
association's movement between the nudged phases, read back through the port's synapses, and the
maps step toward the blocks the scores name: the recall memory's address lesson on the message's
own blocks.

`brain.message.key()` is `cadence.Message`'s key of the same moments (a test holds the two equal),
so the first protocol's retrieval gate R reads the same on it.

## What development found (founders 0 to 5, design input)

Read barred from the meaning, every row taught, unless a line says otherwise.

- **Said from its own message.** `message=280` at 256 association neurons: clause 1.00, held-out
  .92, roles .94 (three founders, 432,528 parameters).
- **The previous-word control** (`history_decays=(0.0,)` in the message's place, 64 association
  neurons): .05, .00, .00.
- **At 64 association neurons** (six founders). The message alone (`message=70`,
  `message_amplitude=0.45`): .83, .61, .43. With the read of one block beside it, the declared
  `read_64`: .89, .85, .77, with 2, 1 and 1 of 6 lives at .90. The failing words are a clause's two
  distractors (.91 and .84 with every previous word given).
- **Beside a free read** (default lessons, six founders; the default composition reads paths .95
  and three hops .40). `message=70` at amplitude .45: paths .74, three hops .01. At amplitude .15:
  paths .95, three hops .04 (three founders). At amplitude .15 with no learned gain
  (`carried_gain=0.0`): paths .83, three hops .11, against .96 and .27 for the default composition
  with no learned gain.
- **Where the cost arises.** The learned gains settle at 2 to 4 on every block. Read with the
  message silent, a brain taught beside it says clauses at .09 to .27 (three founders): it says
  them through the message and the read together. With the fixed quiet message silent at the
  reading, three hops read .22 (.11 with it): the three-hop sentence holds more moments than any
  lesson's message did.
- **Tried and not kept** (none is in the library). The message's drive scaled by what the recall
  read left unfound: the read's greatest attention on a slot that holds a moment is .60 with the
  read free and .42 with it barred, so it does not tell the two apart (paths .53 to .67). The port's
  drive scaled by how far the block read has chosen: .88, .72, .82. A term per step from the block
  the previous read chose: `read_64` with it reads .91, .86, .84 (4, 2 and 1 of 6 lives at .90),
  roles .08 over the read without it at two standard errors of the paired difference and the other
  two panels equal; with it, 140 units a block read .93, .82, .72 and 128 association neurons .91,
  .86, .85. A message held from `GO`, the brain's own words not joining it: paths .26 to .81. Half
  the lesson sentences barred, for one brain that says from either: the default composition itself
  falls to .77 on paths, and at 256 association neurons the brain says clauses at .99, .96, .97
  with the read free and paths at .39, the state-only reading.

## Instrument

`benchmarks/grammar/meaning.py` (`plan --own`, `lives`, `read --own`) and
`benchmarks/grammar/controls.py`. A brain life is `Brain.compose(35, 18, modules=(width,),
seed=founder, backend="cpu", recall=True)` with the arm's settings. Lessons, panels, scores and the
bar are the first protocol's: 4,000 batches of the first grammar from `SeedSequence([founder,
11])`, 64-sentence panels from `SeedSequence([founder, 33])`, whole sentences said closed loop and
greedy; with the bar, the recall slots of the meaning's moments are emptied at `GO` in lessons and
in readings.

## Founders and arms

Founders 9728600 to 9728605 (no life in any lane).

| Arm | The brain | Parameters |
| --- | --- | --- |
| `default` | the default composition, the read free: the reference | 28,716 |
| `heard` | `default` with `history_decays=(0.0,)`, the previous-word signal | 30,992 |
| `barred` | `default` with the read barred: the floor | 28,716 |
| `state_256` | `message=280`, 256 association neurons, every row taught (`learning_surprise=False`), the read barred | 432,528 |
| `state_128` | `state_256` at 128 association neurons | 203,664 |
| `heard_256` | the control of `state_256`: `history_decays=(0.0,)` in the message's place | 153,680 |
| `read_64` | `message=70`, `message_amplitude=0.45`, `message_read=True`, 64 association neurons, every row taught, the read barred | 58,870 |
| `direct_64` | `read_64` without the read of one block | 46,920 |
| `heard_64` | the control of `read_64` | 30,992 |
| `message_free` | `default` with `message=280`, the read free | 101,520 |
| `message_quiet` | `default` with `message=70`, `message_amplitude=0.15`, the read free | 46,920 |
| `message_fixed` | `message_quiet` with `carried_gain=0.0` | 46,914 |
| `default_fixed` | `default` with `carried_gain=0.0`: the reference of `message_fixed` | 28,714 |
| `paraphrase` | `state_256` with `message=140`, `message_phases=8`, taught the first protocol's 6,000 paraphrase batches | 432,532 |
| `paraphrase_heard` | the control of `paraphrase` | 153,680 |
| `paraphrase_read` | `default`, the read free, the same 6,000 batches | 28,716 |
| `capacity_256`, `capacity_128` | `state_256` and `state_128` taught clauses of zero to four distractors alone | 432,528 and 203,664 |
| `capacity_heard` | the control of `capacity_256` | 153,680 |

**Controls.** Every arm that holds a message has its previous-word control beside it: the same
life with `history_decays=(0.0,)` in the message's place. The chamber's LSTM and transformer run at
the parameter counts of `default` (28,398 and 28,658 parameters), `read_64` (58,566 and 58,250) and
`state_256` (430,602 and 432,234), online (one pass in the brain's order) and offline (to the
stopping rule), scored independently on the same panels.

## Readings

Whole sentences on `clause`, `clause_held`, `roles` (both twins), `dyck`, `path`, `clause_long`,
`dyck_deep`, `path_long` (three hops); each word said right by kind with every previous word given
(`said`); on `state_256` and `read_64` the linear probes of every clause component from the
message, the carried trace and the settled association at `GO` and at the last word (5,760 and
1,920 clauses), the context twins intact and with the message emptied, and the two orders of one
meaning; on `message_free`, `message_quiet` and `message_fixed` the same life read with the message
silent and with the read barred; the learned gains; parameters, seconds and analytic
multiply-accumulates.

## Gates

A gate on lives holds where at least 5 of the 6 founders reach its bar.

| Gate | Arm | Reading | Bar |
| --- | --- | --- | --- |
| B, the opt-out | no arm | the recorded-lives tests on the branch; a default life's weights, words and gains hashed on the branch and on its base | every bit equal |
| H1, said from the state | `state_256` | `clause`, `clause_held`, `roles` | .90 each |
| H1w, at the default width | `read_64` | `clause`, `clause_held`, `roles` | .90 each; `direct_64` and `state_128` beside it |
| H2, held | `state_256`, `read_64` | every clause component's probe from the message at `GO` and at the last word | .95 each |
| H2u, used | `state_256`, `read_64` | every clause word kind said right with the previous words given | .95 each |
| H3, one meaning in two orders | `state_256`, `read_64` | the key of the two orders closer than any other meaning's, and both orders said right | .95, .90 |
| H4, paraphrase | `paraphrase` | `para` and `para_held` | .80 each; `paraphrase_read` beside it |
| H5, capacity | `capacity_*` | clauses said exactly per number of distractors | reported: the most distractors said at .90 |
| C, the previous-word control | a message arm less its control, paired by founder | `clause`, `clause_held`, `roles` (`state_256` less `heard_256`, `read_64` less `heard_64`); `para`, `para_held`; each number of distractors | at least .05 and beyond two standard errors; the message counts on a reading only where C holds |
| F, beside a free read | `message_free` and `message_quiet` less `default`, `message_fixed` less `default_fixed`, paired by founder | `path` and `path_long` | no more than .03 under the reference; every panel reported, and `heard` less `default` beside them |

**Expectations, stated before the run.** B holds. H1 holds. H1w fails: `clause` holds and
`clause_held` and `roles` reach .90 on 1 to 3 lives of 6, with `read_64` .2 to .4 over `direct_64`
on both. H2, H2u and H3 hold on `state_256`; on `read_64` H2u fails on a clause's two
distractors. H4 holds on `para` and sits at the bar on `para_held`. C holds on every reading where
the read is barred: the control says no sentence there. F fails on every arm: `message_free` reads
paths .3 to .4 under and three hops near zero, `message_quiet` keeps paths and loses three hops,
`message_fixed` loses .1 to .2 on both. `heard` reads three hops over `default`.

**What a failure names.** Where H1w fails with H2 holding, the meaning is held and the read of one
block does not find it: `said` names the word kind. Where F fails, the life read with the message
silent says whether the lessons or the reading lost the paths, and the life read with the read
barred says whether the brain said its clauses from the message, from the read or from both.

## Left to a later protocol

A learned map from words to concept codes, longer and deeper sentences, persistence through a
second grammar, and the previous-word signal as a default candidate.

## Compute

One AWS c7i.8xlarge, one process per life, one BLAS thread per process.
