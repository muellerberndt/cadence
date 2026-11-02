# Grow and combine brains

Two operations build a larger brain out of a trained one. Both return a new brain and leave the
brains they read unchanged.

- `brain.grow(...)` adds new, empty parts to a trained brain: a recall memory, more read heads,
  a heard history, an efference copy, new output slots.
- `brain.graft(donor, ...)` puts trained regions of another brain into this one: a trained
  visual system into a larger brain, or a whole brain (a measured connectome, say) as one region
  of a larger anatomy.

In both, every synapse that joins the new part to the rest of the brain starts at zero. The
brain therefore acts exactly as it did, and learning then gives those synapses weight.

## Grow a trained brain

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

| Setting | Adds | Starts at zero |
| --- | --- | --- |
| `recall=True`, with any `recall_*` setting `compose` takes | The [recall memory](memory.md#recall-memory): its read ports, empty slots, address and position maps at their founders | The ports' synapses into the association region (and a copy pathway's into the motor region) |
| `recall_heads=k` | More read heads of the brain's recall memory, over its bank | The new ports' synapses |
| `history_decays=(...)` | A heard history (with `history_amplitude`), or the brain's decays followed by new ones | The history's synapses into the association region |
| `efference_amplitude=a` | An efference copy (with `efference_decay`) | The copy's synapses into the association region |
| `slots=(...)` | New output slots after the brain's own: `slots=(4, 2)` gives a brain of one slot of four a second slot of two | Every synapse between the association region and the new motor neurons |

Several parts grow in one call, and a grown brain grows again. The grown brain is the brain's
layout composed again from its recorded construction and its settings, with the same seed, and
the parts added. It carries every learned parameter, the optimizer's history, the plasticity
masks, the memories' records and settings, the actor and critic, the arousal and the random
generators. `grown.describe()` differs from `brain.describe()` in the new parts alone.

## Graft trained regions into a larger brain

### A trained visual system

A composed image brain sees through its `visual` region: the pixels and the feature maps that
read them. What it learns to see lives in the modules above the maps, so a trained visual
system is the `visual` region with the modules a brain trained on images built over it. Graft it
into a larger brain whose regions of the same sizes sit in the same places:

```python
import numpy as np
from cadence import Brain

rng = np.random.default_rng(1)
eye = Brain.compose(inputs=(6, 6), actions=4, modules=(32, 16), seed=0)
for _ in range(20):  # a brain that learns where a bright square lies
    where = rng.integers(0, 4, size=8)
    images = 0.1 * rng.random((8, 6, 6))
    for row, k in enumerate(where):
        images[row, 3 * (k // 2):3 * (k // 2) + 3, 3 * (k % 2):3 * (k % 2) + 3] += 1.0
    eye.lesson(images, where)

larger = Brain.compose(inputs=(6, 6), actions=2, modules=(32, 16, 24), seed=1)
seeing = larger.graft(eye, {"visual": "visual", "module_0": "module_0", "association": "module_1"})
```

The eye's `association` region becomes `module_1` of the larger brain, under a new association
region of 24 neurons. The grafted regions keep the eye's weights. Their joints to the larger
brain's own regions (here `module_1` and the new association) start at zero: at first the graft
settles on what it reads from the image, and the larger brain learns to read the graft.

### A whole brain as one region of a larger anatomy

`into=` grafts every neuron of the donor into one population of this brain, in order. Build the
larger anatomy with the donor's connectome as one of its regions, wire the new regions to it,
name the populations a brain needs, and graft the trained brain in. Here a small developed
layout stands in for a measured connectome such as the fruit fly's:

```python
import numpy as np
from cadence import Brain, Genome, Projection, Region, develop

fly = develop(Genome.composed(6, 4, modules=(16,)), seed=0)  # a stand-in connectome
trained = Brain.compose(6, 4, connectome=fly, seed=0, episodic=False)
rng = np.random.default_rng(2)
for _ in range(10):
    trained.step(rng.random((3, 6)), teacher=rng.integers(0, 4, size=3))

anatomy = develop(
    Genome(
        (Region("fly", circuit=fly), Region("language", 12), Region("words", 5)),
        (
            Projection("fly/association", "language"),  # language and the fly read each other
            Projection("words", "language", reciprocal=False),  # language hears words
        ),
    ),
    seed=1,
)
p = anatomy.populations
anatomy = anatomy.with_populations(
    sensory=list(p["fly/sensory"]) + list(p["words"]), association=p["fly/association"],
    prefrontal=p["fly/prefrontal"], motor=p["fly/motor"],
)
larger = Brain.compose(11, 4, connectome=anatomy, seed=0, episodic=False)
speaking = larger.graft(trained, into="fly")

# with the words silent, the larger brain acts as the fly brain it holds
trained.reset()
x = rng.random((3, 6))
assert np.array_equal(speaking.act(np.hstack([x, np.zeros((3, 5))]), greedy=True),
                      trained.act(x, greedy=True))
```

The language region reads the words, so it is active, and the fly's association moves between
the phases when the fly is taught: the joints between them learn from zero.

### Options

| Option | Meaning |
| --- | --- |
| `regions=["visual", "module_0"]` | Donor populations grafted into this brain's populations of the same names |
| `regions={"association": "module_1"}` | A donor population mapped onto one of this brain's |
| `into="fly"` | The whole donor into one population of this brain, its neurons in order |
| `joints="zero"` (founder) | Every synapse joining the graft to the rest of this brain starts at zero |
| `joints="kept"` | The joints keep this brain's own weights (a newborn's founder draw): the graft is read at once, and the brain changes at once |
| `freeze=True` | The grafted synapses and neurons stop learning; the joints and the rest of the brain learn |

Each grafted pair of populations holds the same number of neurons, taken in order, and every
synapse of the donor among the grafted neurons must be a synapse of this brain: wire the larger
brain as the donor is wired where the graft goes. A synapse of this brain among the grafted
neurons that the donor lacks starts at zero, as a joint does. The grafted neurons take the
donor's biases and gains and the synapses among them the donor's efficacies, both with the
donor's optimizer history. Everything else is this brain's own: its other weights, its memories
and their records, its actor and its arousal.

## How a new part joins the brain

A brain settles to the state in which every neuron's drive balances its activity. A new part
changes that state only through the synapses that join it to the rest. With those at zero (and a
reciprocal pair shares one efficacy, so both directions start at zero), every trained neuron
receives exactly the drive it received before, settles to the same state and makes the same
choice. Where the new part lies inside the layout, the larger graph can sum in another order,
and the trained neurons settle as before within rounding.

The [free/nudged rule](settling.md#learning-the-freenudged-rule) moves a synapse on what its two
neurons do in the two phases, `s+[pre] s+[post] - s0[pre] s0[post]`, never on its own weight. A
synapse at zero therefore learns when one of its neurons is active and the other moves between
the phases:

- a held source (a recall port, the history, the efference copy) is active in both phases, and
  the association region it feeds moves when the teacher nudges the motor neurons;
- a new slot's motor neurons are nudged toward their own labels;
- a graft driven by its own inputs is active, and the regions near the motor neurons move.

A new region with no drive of its own sits at rest, publishes nothing at zero and never learns:
give it inputs or outputs of its own (as the words above), or start its joints with weight
(`joints="kept"`), which changes the brain at once. This is why `grow` refuses a new module or
observer inside the brain's loop: at zero its neurons learn nothing, and at any other start the
brain changes. Compose the larger layout and graft the trained brain into it to choose that
start yourself.

A grafted part settles together with its new brain. At first it settles on what it reads from
its own regions; as the joints grow, it also hears the larger brain, so its states move even
when `freeze=True` holds its weights. A donor region that settled with the donor's own motor
feedback, as an association region does, settles in the larger brain without that feedback.

## Teach and live after the change

- **Teach every slot.** `step` and `lesson` take a label for every slot, and a lesson for a new
  part also moves the synapses the old parts share. Teach the brain's own slots in the same
  lessons, with their correct answers or with the answers the brain gave before it grew.
- **Mind the rates.** New synapses learn at the brain's current rates. A brain whose rates were
  lowered late in training teaches its new parts slowly; raise them with `retune` for a while,
  knowing that the trained parts move faster as well.
- **A new memory starts empty.** Every act writes a slot of a new recall memory; lessons teach
  the brain to read the slots.
- **Living brains.** The arousal, its age included, carries over. A calm adult brain acts
  without learning in `live`, so its new parts learn on aroused moments; teach them with `step`
  or `lesson` to proceed faster.
- **Freezing.** `freeze=True` keeps a graft's weights while the larger brain learns to read it.
  The masks are `brain.learner.plastic_synapses` and `brain.learner.plastic_neurons`; set them
  back to `True` to let the graft learn again.
- **Streams and outcomes.** The new brain starts its streams afresh, as after `reset()`. An
  outcome the trained brain awaits stays with the trained brain.

## Combine the skills of two brains

`graft` combines brains whose regions match. Two brains with different layouts combine through
teaching: give one a new slot and teach it with the other's answers.

```python
import numpy as np
from cadence import Brain

rng = np.random.default_rng(3)
prototypes = rng.normal(size=(12, 8))
first = Brain.compose(inputs=8, actions=4, modules=(32,), seed=3)   # knows kind % 4
second = Brain.compose(inputs=8, actions=3, modules=(16,), seed=4)  # knows kind // 4
for _ in range(30):
    kind = rng.integers(0, 12, size=16)
    x = prototypes[kind] + 0.4 * rng.normal(size=(16, 8))
    first.lesson(x, kind % 4)
    second.lesson(x, kind // 4)

both = first.grow(slots=(4, 3))
for _ in range(30):
    x = prototypes[rng.integers(0, 12, size=16)] + 0.4 * rng.normal(size=(16, 8))
    both.lesson(x, np.stack([first.predict(x), second.predict(x)], axis=1))
```

`first.predict(x)` holds the first slot to what the brain knew; `second.predict(x)` teaches the
new slot what the other brain knows.

## Compatibility

- **Which brains grow.** Every brain `Brain.compose` built that records its construction: a new
  one, or one loaded from a checkpoint that kept the record, as composed brains' checkpoints
  do, those of earlier releases included. A custom anatomy (`compose(connectome=...)`) records
  no construction to compose again; graft it into a larger anatomy instead.
- **Which brains graft.** Any two brains, composed or custom, whose grafted populations match in
  size and wiring, with the same neuron model and the same apical compartment.
- **Devices.** The new brain lives on the backend, device and precision of the brain whose method
  made it.
- **Checkpoints.** A grown or grafted brain is an ordinary brain: it saves and loads like any
  other, and its checkpoint has the format of 0.91.3, which loads it.
- **Determinism.** The same brain grows, and grafts, to the same brain every time.

What does not grow or graft:

- a part inside the brain's loop through `grow` (`modules`, `observers`, `code`, `learned_code`,
  `lateral`, `apical`, other `inputs`); compose the larger layout and graft instead;
- new slots where a memory is laid out per action (`recall_action`, `recall_tokens`,
  `memory_choice`), or where the motor region is a deep brain's [apical](tune.md#deep-brains)
  pool (observers above it): a new neuron would join the pool's normalization;
- a learned code's region, or a joint onto it: the code lives in its tables as well as its
  synapses;
- brains with tied or rated synapses.
