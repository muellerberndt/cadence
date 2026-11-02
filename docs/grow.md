# Grow a brain

`brain.grow(...)` returns a trained brain with new parts: a recall memory it was composed
without, more read heads, a heard history, an efference copy or new output slots. The grown
brain carries everything the brain learned and acts exactly as the brain does; learning then
gives the new parts weight. The trained brain is unchanged.

```python
import numpy as np
from cadence import Brain

rng = np.random.default_rng(0)
brain = Brain.compose(inputs=6, actions=4, modules=(32,), seed=0)
for _ in range(20):
    brain.step(rng.random((4, 6)), teacher=rng.integers(0, 4, size=4))

# the same brain with a recall memory and a second output slot of two actions
bigger = brain.grow(recall=True, slots=(4, 2))

brain.reset()  # the grown brain starts its streams afresh, as after a reset
x = rng.random((4, 6))
assert np.array_equal(bigger.act(x, greedy=True)[:, 0], brain.act(x, greedy=True))

# every lesson labels every slot: the brain's own slot keeps being taught
labels = np.stack([rng.integers(0, 4, size=4), rng.integers(0, 2, size=4)], axis=1)
bigger.step(x, teacher=labels)
```

## The parts

| Setting | Adds | Starts at zero |
| --- | --- | --- |
| `recall=True`, with any `recall_*` setting `compose` takes | The [recall memory](memory.md#recall-memory): its read ports, empty slots, address and position maps at their founders | The ports' synapses into the association region (and a copy pathway's into the motor region) |
| `recall_heads=k` | More read heads of the brain's recall memory, over its bank | The new ports' synapses |
| `history_decays=(...)` | A heard history (with `history_amplitude`), or the brain's decays followed by new ones | The history's synapses into the association region |
| `efference_amplitude=a` | An efference copy (with `efference_decay`) | The copy's synapses into the association region |
| `slots=(...)` | New output slots after the brain's own: `slots=(4, 2)` gives a brain of one slot of four a second slot of two | Every synapse between the association region and the new motor neurons |

Several parts grow in one call. A setting of a part the brain has (`retune` changes those) and
a setting that is not a part are refused.

## What carries over

The grown brain is the brain's layout composed again from its recorded construction and its
settings, with the same seed, and the parts added. It carries:

- every neuron's and synapse's learned parameters, the optimizer's history and the plasticity
  masks;
- the memories' records: the associative store, the episodic records, the recall memory's
  address and position maps, a code's tables;
- the actor and critic, the arousal and the random generators;
- the brain's backend, device and precision.

`grown.describe()` differs from `brain.describe()` in the new parts alone. The streams start
afresh, as after `reset()`: an outcome the brain awaits stays with the trained brain, which
can still receive it.

## Why it acts as before, and still learns

A new synapse onto a neuron the brain already has starts at zero, and so does its reciprocal
partner, which shares its efficacy. The new neurons then add nothing to the drive of any
trained neuron, so every trained neuron settles to the state it settled to before and every
slot the brain had makes the same choice. Where the new part lies inside the layout (new
slots, in front of the memory regions), the larger graph can sum in another order; the trained
neurons then settle as before within rounding.

The [free/nudged rule](settling.md#learning-the-freenudged-rule) moves a synapse on what its
two neurons do in the two phases, never on its own weight, so a synapse at zero learns. A held
source (a recall port, the history, the efference copy) is active in both phases while the
association region moves between them; a new slot's motor neurons are nudged toward their own
labels.

A lesson for a new part moves the synapses the old parts share as well. `step` and `lesson`
take a label for every slot, so the brain's own slots are taught in the same lessons: with
their correct answers, or with the answers the brain gave when it grew.

## What does not grow

- A part inside the brain's loop (`modules`, `observers`, `code`, `learned_code`, `lateral`,
  `apical`, other `inputs`): its neurons would sit in reciprocal pairs on both sides, where a
  start at zero gives every new neuron the same updates and any other start changes the brain
  at once. Compose a new brain for these.
- A custom anatomy (`compose(connectome=...)`), which records no construction to compose again.
- New slots where a memory is laid out per action (`recall_action`, `recall_tokens`,
  `memory_choice`), or where the motor region is a deep brain's [apical](tune.md#deep-brains)
  pool (observers above it): a new neuron would join the pool's normalization.
