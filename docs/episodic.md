# Episodic memory: what happened in this situation before

A brain's synapses learn slowly. Each update moves the whole policy a little, and neighbouring
situations share most of their representation, so one lesson barely moves one answer. Animals
also learn in one go: a place where food was found once, a route a parent showed once, the
step of a song that comes next. Since 0.83.0 the composed brain can keep such episodes in its
hippocampus, an episodic memory beside the slow cortex whose records vote in every choice:
`Brain.compose(..., memory_choice=True)`. It is not yet the default; [when to turn it
on](#when-to-turn-it-on) says why.

## How it works

```text
observation now ─────────────────────► place records ───┐
seen and done before ─┐                                  ├─► mean ─► vote ─┐
observation now ──────┴─► temporal context ─► context records ─┘           ├─► softmax ─► action
observation now ─► settle ─► motor readout ─────────────────────────────────┘

outcome, when known ─────────────► record of the chosen action under its key
reward beyond what was expected ─► replay: the decisions on the recent path, backwards
```

- **Two keys for every record.** Each record is kept twice. The place records are keyed by
  the present observation. The context records are keyed by the temporal context: the
  present observation together with a fading trace of what the stream saw and did before,
  past observations and the brain's own issued actions. `memory_context` is the share the
  past keeps from one moment to the next; the present observation enters with the rest.
- **Why two.** Two moments that look alike but follow different histories get different
  context keys, so a cue that is gone still shapes the vote, and so does the place in a
  sequence of the brain's own actions. Where history does not matter, the context splits one
  situation's records over every path that reaches it; the place records keep them together,
  so a known situation reached along a new path is still recognized. The trace is fixed,
  not learned: a record stays addressable while the cortex learns.
- **Records** hold one value per action for each key, in reward units. Keys are separated
  first (a fixed random expansion to 1,024 cells, of which the 24 most active stay), so
  different situations write different synapses.
- **What is written.** The outcome of the chosen action, when it is known: a reward or a
  punishment, the end of an attempt (`done`), a surprise that woke a calm `live` brain, or an
  outcome the caller marks with a positive `salience=`. A zero outcome in the middle of an
  attempt is not yet an outcome and leaves the record as it was. A caregiver's demonstration
  (`teacher=`) is recorded for the situation it was shown in, at `memory_demonstrations`, as
  if the shown action had paid that much.
- **Replay.** A reward replays the stream's recent path backwards, up to `memory_replay`
  decisions. The path keeps one decision per situation, the one that left it: an action that
  left the situation unchanged is not on the path and is not credited. Replay runs through the
  decisions whose own outcome was nothing and stops at the last decision whose outcome was
  known, a reward or a punishment: outcomes bound the episode, so nothing has to mark the end
  of an attempt. Each replayed decision gains credit discounted by the actor's `gamma` per
  step, up to the discounted reward, and replay only ever adds credit.
  - A reward that ends an attempt (`done=True`) closes an episode: it is replayed whole, into
    both kinds of record.
  - In a continuing life it is not known where the paid effort began, so a reward is replayed
    in proportion to its surprise, the reward less what the memory expected of the decision
    it paid; an expected reward replays nothing. Each decision gains it in its context
    record, that decision in the history it was taken in, and in its place record only as far
    as it was an exploration rather than the brain's best guess: a habitual choice that
    happened to come before a reward says nothing about what paid apart from its history, an
    exploration that came before a surprising reward says much.
- **The vote.** At every choice, greedy or sampled, the mean of the place and the context
  records is added to the motor readout before the softmax. The slow cortex keeps learning
  from the actions the brain takes, so what the memory chose and what paid is consolidated
  into the synapses over time.
- **Two speeds.** The store has the per-stream fast residual and the shared consolidated
  matrix of [`SynapticMemory`](memory.md#memory-in-the-composed-brain). A write corrects the
  stream's record halfway at once (`memory_rate` 0.5), and that correction fades by
  `memory_decay` per later write; the record every stream shares, and that a reset keeps,
  moves `consolidation` (0.05) of the way, faster with a supplied `salience=`. One outcome is
  remembered at once; repetition makes a record lasting, and one misleading outcome cannot
  erase what repetition established.

## Turning it on

```python
from cadence import Brain

brain = Brain.compose(inputs=3, actions=2, arousal=True, memory_choice=True)
brain.step([[1.0, 0.0, 0.0]], teacher=[0])                  # shown once: remembered
action = brain.live([[1.0, 0.0, 0.0]])                      # the record votes for 0
```

`memory_choice=True` brings the settings below. Each can be passed beside it.

| Setting | With `memory_choice=True` | What it does |
| --- | --- | --- |
| `memory_choice` | `True` | The records vote in the choice; the founder `False` keeps the hippocampus of earlier releases, keyed by the senses and read into the motor drive |
| `memory_context` | `0.5` | The share of the past in the context key; 0 keeps the place records alone |
| `memory_replay` | `8` | How many decisions back a surprising reward is replayed |
| `memory_demonstrations` | `1.0` | The value at which a shown action is recorded; 0 records none |
| `memory_rate` | `0.5` | How far one write moves the stream's record at once; 1.0 lets the last outcome replace it |
| `memory_expansion`, `memory_winners` | `1024`, `24` | Separation of the keys; 0 cells use the key as it is |

These are construction settings: pass them to `Brain.compose(...)`; `retune` refuses them,
except `memory_rate`, a store setting like `memory_decay`, `memory_amplitude` and
`consolidation`, which `retune` changes for both stores together. The same settings turn it
on in `Brain(...)` and `Brain.build(...)`, where every setting has to be named.

## When to turn it on

Turn it on when a skill is taught by showing or correcting, when a chain of actions is paid only
at its end, when a cue that is gone must still decide an answer, and for sequences of the
brain's own actions. Nothing has to mark the end of an attempt. Measured on fresh founders and
seeds against the composition without it (the default `Brain.compose`), none of them used to
choose the settings:

| Task (fresh founders or seeds) | Without | With `memory_choice=True` |
| --- | --- | --- |
| Four-step crafting chain, a caregiver corrects executed mistakes (5 founders): success after teaching | 38.1% | 98.7% |
| the same, then 80 episodes alone through `live` | 9.0% | 97.8% |
| the same chain from reward alone, through `step`: after training, then alone | 9.1%, 9.5% | 73.2%, 88.0% |
| A second crafting chain (5 seeds): taught, after the caregiver leaves | 2.8% | 99.0% |
| the same from reward alone, after the caregiver leaves | 15.0% | 75.6% |
| A four-step chain in one continuing life, paid only at its completion, no `done` (10 seeds): greedy completions per thousand moments | 0 | 108 |
| Rules that reverse in a continuing life (odour nursery, 20 lives): lives within every gate; reversal lags | 20 of 20; 46, 57 | 20 of 20; 55, 52 |
| Delayed response learned from reward (6 seeds) | | 6 of 6 at 1.0 |
| Two four-note songs cued once, 10 lessons (6 seeds): notes right | 0.52 | 0.65 |
| Taught recall of a vanished cue (4 founders): clean-4, clean-8, distractor-2, capacity-2 | 0.70 | 0.75 |
| Continuing bandit whose history does not matter (6 seeds): hit rate | 0.86 | 0.75 |

Where it does not help yet: on the bandit whose history does not matter it scored below the
composition without it; on the continuing chain, four of the ten seeds completed no chain with
or without it; the recall gain is uneven, +0.31 on one founder and -0.20 on another; and after
80 episodes alone through `step`, the frozen greedy policy of the corrected chain can stall at
one stage while the sampled policy still succeeds (through `live` it does not). It is not the
default for now.

On the [Stone Age environment](https://github.com/ali12mahmood12345-boop/cadence-stone-age)'s
own benchmarks of its four-step crafting chain (seeds 42 to 46, frozen full-chain success
after training): backward lessons stopped at competency go from 8.4% to 25.6% (the untrained
founder 9.0%, uniform random 7.2%) with a third of the lesson trips and a fifth of the rescue
trips, and retention rescue goes from 6.2% to 37.4%, keeping all four steps where without it
two of them collapse.

## Settings

| Setting | Raise it when | Lower it when |
| --- | --- | --- |
| `memory_context` | the answer depends on what came long before: a cue that is gone, the next step of a long sequence of the brain's own actions | history does not matter, and the context records only dilute the place records (0 keeps the place records alone) |
| `memory_replay` | rewards come at the end of long chains of decisions | (0 replays nothing) |
| `memory_demonstrations` | shown actions should outweigh the rewards of the brain's own trials | the caregiver is unreliable (0 records no demonstration) |
| `memory_expansion` | many situations must be told apart | (0 uses the key unseparated) |
| `memory_rate` | the world changes quickly and the latest outcome is the best guide | outcomes are noisy, or one failure should not undo a learned skill |

Where the world has attempts (a chain completed, a trial over, a game lost), `done=True` makes
the outcome of the last decision known and starts the next attempt with a fresh context and
path. The memory needs no such mark to learn a sequence: rewards and punishments bound what a
reward replays.

## Check that the memory does the work

Compare the same acquired brain with and without its records, on copies:

```python
import os
import tempfile

path = os.path.join(tempfile.mkdtemp(), "brain.npz")
brain.save(path)
remembering, forgetting = Brain.load(path), Brain.load(path)
forgetting.hippocampus.clear()  # erase the place records
forgetting.context_memory.clear()  # and the context records
x = [[1.0, 0.0, 0.0]]
print(remembering.act(x, greedy=True), forgetting.act(x, greedy=True))
```

If the answers differ, the records decide that choice. If they never differ, the slow
cortex has consolidated what the memory held, or the memory holds nothing for that key.

## What it is not

- It is not a list of episodes. Its storage is fixed by the key width and the separation; it
  does not grow with experience. Correlated keys interfere, as every delta-rule store does.
- The keys are not learned. The memory recalls what happened in the same place and in the
  same context; it does not discover which part of the history matters. Where it matters,
  the place records of the aliased moments average over histories and halve the context's
  vote; where it does not, the context records add little.
- Records are in reward units. With rewards of a very different size, the vote weighs more or
  less against the readout; `memory_amplitude` scales the read.
- `predict` and `fit` answer independent samples and read no memory.

See [context](context.md) for the carried interpretation of the working trace,
[memory](memory.md) for the stores and their rules, and [sequential tasks](sequential-tasks.md)
for chains of actions.
