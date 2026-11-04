# Your own anatomy

Use this page for your own regions, a measured connectome or evolved wiring. A `Genome` lists
regions and projections, `develop` lays them out into a `Connectome`, and
`Brain.compose(inputs, actions, connectome=...)` gives the anatomy the composed founders.

## Genome, development, brain

```python
import cadence as cd
from cadence.regions import cortex, motor_cortex

genome = cd.Genome(
    regions=(cd.Region("senses", 5), cortex(16), motor_cortex(3)),
    projections=(
        cd.Projection("senses", "association", reciprocal=False),
        cd.Projection("association", "motor"),
    ),
)
network = cd.NeuralGraph(cd.develop(genome, seed=0), cd.learning_neuron_model())
```

This graph has no motor competition (`motor_cortex` defaults to `lateral=0.0`) and no
Brain loop. Wrap it in a brain in one of two ways:

- `Brain.compose(inputs, actions, connectome=connectome, ...)` adds the composed founders
  ([defaults](tune.md#defaults-and-expert-overrides)), and the episodic founders under
  `memory_choice=True`. Its `sensory` (or `visual/input`) population holds the inputs (the
  code's units with `code=units`) and its `motor` population the actions. The anatomy brings
  its own layout, so the layout settings (`modules`, `observers`, `lateral`, `scales`,
  `features`, `field`) stay at their founders. To change the composed layout itself, edit or
  evolve `Genome.composed(inputs, actions, ...)`, the layout `compose` develops
  ([mutation](#mutation)).
- `Brain(connectome, ...)` wraps any connectome with `association`, `motor` and `sensory`
  (or `visual/input`) populations and takes the same constructor options; `inputs=` is the
  observation width when the sensory population is a `code=units` code. It keeps the
  composed learning, actor, arousal and memory founders. With `memory_choice=True` it starts
  the episodic context, replay, demonstrations and separation at zero unless you pass them
  ([episodic memory](memory.md#episodic-memory-what-happened-in-this-situation-before)).

`Genome.observation_width(inputs, code=None, learned_code=None)` is the key width the
memories use, and so the size of a custom anatomy's `recall` and `next` regions.

[Write a cortex](#write-a-cortex) covers regions, ports and projections;
[evolve a brain](#evolve-a-brain) covers genomes and selection.

## Write a cortex

A cortex is a named group of neurons with declared projections to other cortices. In a
`Genome` it is a `Region`; `develop` lays every region into one connectome, where each
becomes a named population, a port. A learning head is a `Learner` that owns the synapses
between some ports and changes them from settled phases. All regions settle in one
equilibrium. [Compose a brain](tune.md) builds a ready brain from these parts; this section
wires one by hand.

### Blank and designed regions

`Region(name, size)` is a blank region, a group of neurons with no synapses of its own. The
genome's projections connect it, and `mutate` may change its size.
`Region(name, circuit=connectome, inputs=None, outputs=None)` is a designed region. It
carries its own synapses and named populations, takes its size from the circuit and keeps
that size under mutation. `inputs` names the circuit population that receives projections
and `outputs` the population that sends them; each defaults to the whole region. A region
name is nonempty and holds no `/`; a blank region refuses `inputs` and `outputs`.

```python
import numpy as np
import cadence as cd

ring = cd.Connectome.from_synapses(
    6, pre=[0, 1, 2, 3, 4, 5], post=[1, 2, 3, 4, 5, 0], sign=[0.5] * 6,
    populations={"cue": [0], "readout": [3]}, label="ring",
)
loop = cd.Region("loop", circuit=ring, inputs="cue", outputs="readout")
print(loop.designed, loop.size, loop.neurons("readout"))   # True 6 (3,)
```

`designed` is `True` for a region with a circuit. `neurons(population=None)` returns
region-local indices, of one circuit population or of every neuron. `to_dict()` gives the
name and size, plus a designed region's circuit label, digest, `inputs` and `outputs`.

### The catalogue

`cadence.regions` builds the standard cortices, engineering analogues named for their
intended functions. A learner over a slotted motor port takes the same `slots`, one softmax
choice per slot.

| Builder | Region | Populations and synapses |
|---|---|---|
| `visual_cortex(height, width, *, channels=1, features=8, field=3, stride=1, init=1.0, seed=0, name="visual")` | designed | `input`: one neuron per pixel and channel, in the order of a `(height, width, channels)` image flattened row by row. `output`: `features` maps; each neuron reads one `field` by `field` window of the input, the windows stepping by `stride`. `field` fits within the image. Efficacies start with random signs and fan-scaled magnitudes times `init` (nonnegative). |
| `cortex(size, *, lateral=0.0, name="association")` | blank; designed for a nonzero `lateral` | With a nonzero `lateral`, every ordered pair of its neurons carries a synapse of that weight; a negative weight makes the neurons compete. |
| `motor_cortex(actions, *, lateral=0.0, name="motor", slots=1)` | designed | `actions`: one neuron per action, with a synapse of weight `lateral` between every ordered pair when `lateral` is nonzero. `slots` splits the actions into separate competitions that settle together: a count that divides `actions`, or one size per slot. The lateral synapses stay within each slot. |
| `prefrontal_cortex(holds, *, name="prefrontal")` | blank | One neuron per neuron of `holds` (a `Region` or a size), for a `Trace` of the held region that drives it as a stimulus. Keep the sizes tied under evolution with `mutate(..., tied=(("association", "prefrontal"),))`, held region first. |

```python
from cadence.regions import cortex, motor_cortex, prefrontal_cortex, visual_cortex

retina = visual_cortex(8, 8, features=4)          # 64 pixels, then 4 maps of 6 by 6
association = cortex(24)
motor = motor_cortex(3, lateral=-0.5)
held = prefrontal_cortex(association)
print(retina.size, retina.inputs, retina.outputs)                  # 208 input output
print(association.designed, motor.designed, held.name, held.size)  # False True prefrontal 24
print(motor_cortex(4, lateral=-0.5, slots=2).circuit.synapses)     # 4: two pairs per slot
```

### Projections

`Projection(pre, post, density=1.0, sign=0.0, scale=1.0, count=1.0, reciprocal=True)`
declares synapses from the neurons of `pre` to the neurons of `post`. An end names a
region, meaning its `outputs` population when it sends and its `inputs` population when it
receives (every neuron of a blank region), or one population as `region/population`.

| Field | Meaning |
|---|---|
| `density` | Probability in `[0, 1]` that a pair of neurons, one from each end, carries a synapse. |
| `sign` | Mean sign in `[-1, 1]`: each synapse is excitatory with probability `(1 + sign) / 2`. |
| `scale` | Nonnegative; multiplies the magnitudes, drawn uniformly from `[0, sqrt(6 / (pre neurons + post neurons))]`. |
| `count` | Synaptic contacts per synapse, nonnegative. |
| `reciprocal` | `True` adds the reverse of every synapse with the same signed weight, so a learner ties the two into one reciprocal synapse and a nudge travels back. `False` declares a one-way projection. |

Development draws every magnitude and sign from its seed. Synapses that two projections
place on the same ordered pair of neurons merge into one, whose count is the sum and whose
sign is the count-weighted mean; a synapse from a neuron onto itself is dropped. `Genome(regions, projections, label="genome")`
checks every projection end when it is built and refuses an empty or duplicated region name list.

### The apical compartment

Set `Brain.compose(..., apical=fraction)` for a chain of several reciprocal processing
regions. Without it, each region hears the region above through the reverse half of the pair
that carries its own output up. Summed with the bottom-up input, that top-down input closes a
loop whose gain grows with the weights. Near the critical gain the settle slows and the region
carries an echo of the answer instead of a representation of its input, so a second module adds
little to the first.

The compartment lands the top-down half of every reciprocal pair between two regions apically,
as the areas above a cortical pyramidal cell reach its apical tuft. The regions are the
populations named without a slash, in the order `develop` lays them out. A synapse is apical
when its reverse synapse exists and its presynaptic region comes after its postsynaptic one: a
module's input from the module above, the association's from the motor region, any region's
from an observer. One-way projections (the sensory input, the working trace, the efference
copy, the history and the recall memory) stay basal. In every sweep, a region with apical input
`u` adds to every member's basal input `b` (its other synapses, the stimulus, the bias, a
nudge) the drive

```text
fraction * rms(b) * u / sqrt(rms(u)^2 + rms(b)^2)
```

with the root mean squares taken over the region, row by row. The top-down direction is kept,
and its magnitude stays below `fraction` times the region's bottom-up drive however large the
weights grow.

- A nudged settle holds the gain each pool had at the free phase's state, one per row and pool
  (`Brain.apical_gain`, carried as `Nudge.apical_gain`). Where the drive sits at its bound, the
  per-sweep normalization would keep only the direction of `u` and suppress a nudge's change
  along it; at the held gain the change arrives undistorted. The free phase and its fixed
  point are the normalized ones.
- A brain with the compartment settles at half the founder's step, which damps the loop's fast
  mode and moves no fixed point.
- `NeuralGraph(..., apical=Apical(synapses, groups, fraction))` gives any graph the
  compartment: `synapses` a bool per synapse, `groups` a pool per neuron.

Teach such a brain at `teaching="deep"`. Credit that reaches a region k levels below the motor
region shrinks by about the fraction per level, below a coarse settle's error. The deep point is
the normalized point with the [projection-relative floor](teach.md#rates-under-normalization)
at 0.1 and nudged phases of a whole 32 sweeps from a free phase at the founder's tolerance (the
opposite phases' common drift cancels in the contrast). The founder of the
[depth benchmark](../benchmarks/depth/README.md) composes `apical=0.25` with
`teaching="deep"`. On its task, which takes two layers of thresholds, one to four processing
regions order by depth, and four finish about 2.2 percentage points below a network of two
hidden layers ([deep brains](tune.md#deep-brains)).

### Ports after development

`develop(genome, seed=0)` lays the regions out in order as contiguous populations named
after them, adds each designed region's synapses and its populations as
`region/population`, and draws every projection. The result is one `Connectome` for one
`NeuralGraph`, identical for the same seed. A port is a named population of that connectome.

```python
genome = cd.Genome(
    regions=(cd.Region("senses", 6), association, motor, cd.Region("value", 1), loop),
    projections=(
        cd.Projection("senses", "association", density=0.5, reciprocal=False),
        cd.Projection("association", "motor"),
        cd.Projection("association", "value", scale=0.5),
        cd.Projection("loop", "association", sign=1.0, reciprocal=False),
        cd.Projection("association", "loop/cue", sign=-1.0, density=0.25, reciprocal=False),
    ),
    label="two-heads",
)
connectome = cd.develop(genome, seed=0)
print(list(connectome.populations))   # eight ports, motor/actions and loop/cue among them

def between(connectome, pre, post):
    """The synapses from the neurons of port ``pre`` to the neurons of port ``post``."""
    return np.isin(connectome.pre, connectome.populations[pre]) & np.isin(
        connectome.post, connectome.populations[post]
    )

print(between(connectome, "senses", "association").sum())   # 64 of 144 pairs at density 0.5
print(between(connectome, "association", "senses").sum())   # 0: a one-way projection
print(between(connectome, "motor", "association").sum())    # 72: the reverse synapses
print((connectome.sign[between(connectome, "loop", "association")] > 0).all())  # True
```

<a id="genome-records"></a>
`genome.region(name)` returns one region. `Genome.to_dict()` is a plain record of the
regions, projections and label, the form a lineage keeps. `Genome.from_dict(record,
designed=None)` reads it back. The record holds a designed region's circuit label and digest
only, so `designed` supplies each designed region by name, and its digest, size, `inputs` and
`outputs` must match the record.

```python
record = genome.to_dict()
again = cd.Genome.from_dict(record, designed={"motor": motor, "loop": loop})
assert cd.develop(again, seed=0).digest() == connectome.digest()
```

### Which synapses a head owns

A learning head is a `Learner` with two boolean masks: `plastic_synapses`, one entry per
synapse, and `plastic_neurons`, one entry per neuron whose bias moves. Synapses and biases
outside the masks keep their values through the head's updates, decay and clipping. The
helpers below build the masks from port pairs.

A reciprocal pair shares one efficacy. The learner zeroes the steps of frozen synapses,
averages the two directions of every pair and zeroes the frozen steps again, so a synapse
whose reverse is frozen moves by half its own step. A head owns both directions of every
reciprocal projection it learns. `Learner(synapse_rate=...)` multiplies each synapse's step
before that averaging, so one head learns its projections at different rates.

```python
def head(connectome, *pairs):
    """Both directions of the synapses between each pair of ports."""
    owned = np.zeros(connectome.synapses, bool)
    for pre, post in pairs:
        owned |= between(connectome, pre, post) | between(connectome, post, pre)
    return owned

def ports(connectome, *names):
    """A neuron mask of the named ports."""
    mask = np.zeros(connectome.n, bool)
    for name in names:
        mask[list(connectome.populations[name])] = True
    return mask
```

### A learner over one head

Heads with disjoint masks share one graph. Build the first head on the `NeuralGraph` and each
further head with `Learner.sharing(first, outputs, **options)`, where `options` are the
`Learner` fields after `outputs`. The heads share one holder, so each update replaces the graph
every head reads ([two learners on one graph](settling.md#two-learners-on-one-graph)). Disjoint
masks also keep one head's updates off the other's synapses.

`Learner.step(drive, labels)` settles a free phase, the nudged phases toward the labelled
outputs and one update. The phase operations `free`, `nudged` and `update` take an explicit
target pattern, here a value for the quadratic head.

```python
brain = cd.NeuralGraph(connectome, cd.learning_neuron_model())
policy = cd.Learner(
    brain,
    connectome.populations["motor/actions"],
    plastic_synapses=head(connectome, ("association", "motor")),
    plastic_neurons=ports(connectome, "motor"),
)
value = cd.Learner.sharing(
    policy,
    connectome.populations["value"],
    config=cd.LearnerConfig(nudge="quadratic"),
    plastic_synapses=head(connectome, ("association", "value")),
    plastic_neurons=ports(connectome, "value"),
)
assert not (policy.plastic_synapses & value.plastic_synapses).any()

drive = np.zeros((2, connectome.n))
drive[:, list(connectome.populations["senses"])] = np.eye(6)[:2]
before = brain.efficacy.copy()

policy.step(drive, np.array([0, 2]))          # one action index per row
assert value.brain is policy.brain            # the value head settles the updated graph
target = np.zeros((2, connectome.n))
target[:, list(connectome.populations["value"])] = [[1.0], [0.0]]
free = value.free(drive)
plus = value.nudged(drive, free, target)
minus = value.nudged(drive, free, target, sign=-1.0)
value.update(free, plus, minus)

moved = policy.brain.efficacy != before
owned = policy.plastic_synapses | value.plastic_synapses
assert moved[owned].any() and not moved[~owned].any()
```

### Settle or record

Use a settling cortex when the answer depends on a joint state of many neurons: completing a
missing part of the reading, carrying context across a delay through a recurrent state or a
`prefrontal_cortex`, or learning a policy under late reward through `ActorCritic`
([reward](reward.md)). Use a delta-rule store outside the connectome (`FastSynapses`,
`SynapticMemory`) when the outcome follows the reading it belongs to and the reading holds
what decides it. A store is read and written without a settling step and touches no
`NeuralGraph` parameter ([a store or a settling learner](memory.md#a-store-or-a-settling-learner)).
A store's key can include the activation of a port, so a store and a settled policy combine in
one brain; [Compose a brain](tune.md) builds both.

## Brains from a connectome

A measured wiring diagram becomes one `NeuralGraph`: every neuron a unit of the graded rate
model, every synapse class one entry with its contact count and sign, one global gain. The
steps below are the import recipe. Biological dynamics and behaviour need their own validation
([numerical and learning contracts](contracts.md)). The
[Connectome demo](https://github.com/muellerberndt/cadence-demos/tree/main/connectome) runs
compiled fly wiring with plastic connections in the browser.

To add parts to a trained brain on a connectome, build a larger anatomy with the connectome as
one region (`Region(name, circuit=connectome)`) and the new regions wired to it, compose it,
and graft the trained brain in with `into=`. The joints start at zero, so the larger brain
settles on the trained weights as the trained brain does, with its own memories and settings,
until the joints learn
([a whole brain as one region](grow.md#a-whole-brain-as-one-region-of-a-larger-anatomy)).

### 1. Custody: manifest and dictionary decisions

Pin the release files by URL and SHA-256, fetch them once, and build a fixture that the checks
rebuild from the same bytes. Commit a manifest (counts, hashes, `Connectome.digest()`, the
dictionary) and rebuild the fixture itself with one command. The dictionary records every
import decision in one place:

- **Which objects are neurons.** For the fly, the proofread objects of every super class except
  glia, trachea and non-neuronal: 150,802 of 188,508.
- **The synapse floor** (`min_count`, section 2), and the seams it must leave untouched.
- **The sign of every class**, from the presynaptic transmitter: acetylcholine +1; GABA,
  glutamate and histamine -1; modulators 0. Transmitter predictions are per neuron and
  imperfect, so the sign map is a declaration.
- **What the release lacks.** A release of chemical synapses only has no electrical paths, and
  no fact about them can be asked.

### 2. The connectome and its populations

`Connectome.from_synapses(n, pre=, post=, count=, sign=, populations=, label=, min_count=)`
drops self-synapses and every input row with fewer than `min_count` contacts (the floor applies
before merging), merges parallel rows between the same pair by summing counts (the sign becomes
the count-weighted mean), and sorts synapses by receiving neuron, then sender. `count` defaults
to 1, `sign` to +1, `label` to `"connectome"`, `min_count` to 0. It refuses indices outside
`[0, n)`, negative counts (inhibition goes in `sign`), non-finite values and a negative or
non-finite `min_count`. `populations` maps a name to neuron indices (stored sorted and
deduplicated). Names are the interface of every later tool (protocols, `seam_report`,
`calibrate_bias`), so name every set a gate or a page will read: afferents by modality and
side, motor neurons by muscle and side, the memory site's cells by type.

The floor removes noise and also removes distributed memories, which are many weak synapses.
`seam_report(connectome, pre, post)` says what custody left of a seam before a lesson is
designed on it, and the seam a lesson will move is kept at every count:

```python
import numpy as np
from cadence import Connectome, NeuralGraph, NeuronModel, seam_report

rng = np.random.default_rng(0)
groups = {"orn:fruit": range(0, 20), "orn:yeast": range(20, 40), "ln": range(40, 70),
          "pn:fruit": range(70, 90), "pn:yeast": range(90, 110), "kc": range(110, 290),
          "mbon": range(290, 292)}

def wire(a, b, p, mean):  # random classes from population a to b, geometric counts
    a, b = np.array(groups[a]), np.array(groups[b])
    i, j = np.nonzero(rng.random((len(a), len(b))) < p)
    return a[i], b[j], rng.geometric(1 / mean, len(i))

parts = [wire("orn:fruit", "pn:fruit", 0.4, 6), wire("orn:yeast", "pn:yeast", 0.4, 6),
         wire("orn:fruit", "ln", 0.3, 4), wire("orn:yeast", "ln", 0.3, 4),
         wire("ln", "ln", 0.5, 6), wire("ln", "pn:fruit", 0.5, 4), wire("ln", "pn:yeast", 0.5, 4),
         wire("pn:fruit", "kc", 0.03, 6), wire("pn:yeast", "kc", 0.03, 6),
         wire("kc", "mbon", 0.5, 1.5)]
pre, post, count = (np.concatenate(x) for x in zip(*parts))
sign = np.ones(len(pre))  # all cholinergic in this toy

floored = Connectome.from_synapses(292, pre=pre, post=post, count=count, sign=sign,
                                   populations=groups, min_count=2)
print(seam_report(floored, "kc", "mbon")["coverage_pre"])  # about 0.28: the floor thinned it

seam = np.isin(pre, groups["kc"]) & np.isin(post, groups["mbon"])
keep = (count >= 2) | seam  # the floor everywhere except the seam
C = Connectome.from_synapses(292, pre=pre[keep], post=post[keep], count=count[keep],
                             sign=sign[keep], populations=groups, label="toy-lobe")
report = seam_report(C, "kc", "mbon")
assert report["coverage_pre"] > 0.7
```

`seam_report` returns `classes`, `synapses`, `coverage_pre` (the share of senders with a class
onto some receiver), `classes_per_post` and `median_count`. Reading it, with `preflight` and
`naive_efficacy`, is described in [learning from reward](reward.md#check-before-the-first-lesson).
The default `cpu` backend keeps a large connectome sparse; never call `brain.dense()` on one.

### 3. The gain by protocol

Select the global gain (`NeuronModel(gain=)`) with `select_gain` from two or three training
facts from the physiology literature, and score the held-out rest. Each shuffled wiring (same
neurons and counts, receiving endpoints permuted) selects its own gain as the control. A mean
over a large readout cannot reach 0.5, so large readouts need the `sparse` predicate, and
left-right sets need `lateralized`. Write the facts with their sources before the run. The
[protocol instrument](../benchmarks/instruments/README.md) owns `Protocol`, the predicates,
[the shuffled control](../benchmarks/instruments/README.md#the-shuffled-control) and
[gain selection under a sparsity cap](../benchmarks/instruments/README.md#gain-selection-under-a-sparsity-cap); it is a
module of a source checkout, imported with `benchmarks/instruments` on `sys.path`.

### 4. A gain per cell class

A loop of excitatory local neurons that is supercritical at the steering circuit's gain lights
every odour into the same code, and raising their threshold leaves a multiplicative loop
ignited. `NeuralGraph(log_gain=)` holds one log gain per neuron on its
outgoing synapses (weight = `gain * count * efficacy * exp(log_gain[pre])`), and `select_gain`
picks a class's value the same way as the global gain:

```python
from protocol import Levels, Protocol, select_gain   # benchmarks/instruments on sys.path

def make(attenuation):  # the candidate: minus the log gain of the local neurons
    log_gain = np.zeros(C.n)
    log_gain[list(C.populations["ln"])] = -attenuation
    return NeuralGraph(C, NeuronModel(gain=0.05), log_gain=log_gain)

protocol = Protocol(
    stimuli={"fruit": ("orn:fruit",), "yeast": ("orn:yeast",)}, rows=[],
    training=[("fruit", "kc", "sparse"), ("yeast", "kc", "sparse"),
              ("fruit", "kc", "specific", "yeast")],
    levels=Levels(sparse_min=0.02, sparse_max=0.15, specific_max=0.25, code_level=0.05))
attenuation, table = select_gain(make, protocol, [0, 0.7, 1.4, 2.1, 3.0], sparsity_cap=0.3)
print(attenuation, [(row["gain"], row["facts_passed"]) for row in table])
assert attenuation == 0.7 and not table[0]["admissible"]
brain = make(attenuation)
```

`select_gain` keeps the candidates that are admissible (at most `sparsity_cap` of all neurons
active under every training stimulus; the default cap is 0.05 and `None` turns it off), takes
the one that passes the most training facts, and breaks ties by the smallest value. It raises
`ValueError` on an empty grid or when no candidate is admissible. In the example the
unattenuated lobe activates 0.32 of all neurons, over the cap, and 0.7 is the smallest value
that passes all three facts.

Select on the whole brain, because a sub-net's own stimulated receptors can exceed the cap, and
declare the value in the dictionary next to the global gain. Measure the code at the learning
site before designing a lesson; a code shared by every stimulus carries nothing a lesson can
attach to.

### 5. What a rate model cannot carry

Timing-coded pathways, electrical synapses, neuromodulation, hormones and peptides are outside
the model, and transmitter signs are a fast-interaction map only. Declare such a pathway, supply
it from outside (a hand-written layer or the body's own loop), and keep the boundary visible
on any page that lets the brain override it. The fly's wingbeat-timescale equilibrium reflex
rests on haltere spike timing, so a closed loop supplies it.

### 6. Graph coverage in a browser

Export the full graph the dictionary selected. A reduced graph is an approximation that needs
its own evidence: equal population means do not give equal member-level behaviour, so compare
individual activities, readouts and task outcomes under the same stimuli. Export synapses in
`Connectome`'s order (receiving neuron, then sender) so a member-level parity test can hold a
browser engine to the library.

### 7. Before the first lesson

The learning rule is the same actor-critic as on any brain ([learning from reward](reward.md)).
Decide which neurons read the action and which synapses are plastic (the seam the animal is known
to change, passed as the learner's `plastic_synapses` mask). Declare the two operating points the
connectome does not give:

- The readout's threshold. A readout cell on a rail gives a nudge little to act on;
  [`calibrate_bias`](settling.md#calibrating-the-operating-point) searches its bias into
  `NeuralGraph(bias=)`.
- The seam's starting strength. `naive_efficacy` gives plastic classes equal magnitude into
  `NeuralGraph(efficacy=)`, a controlled initialization that changes the measured relative
  strengths; report it as such.

Run `preflight` before the first decision
([check before the first lesson](reward.md#check-before-the-first-lesson)); an empty warning list
does not certify equilibrium or useful acquisition. The assay then gives every action an
outcome, one decision per episode credited to that decision.

## Evolve a brain

Evolution changes a genome between lives. `mutate` draws an offspring, and `evolve` keeps
the genomes whose lives score best under a fitness the application supplies. A `Genome`
decides which regions a brain has, how large they are and how they project; any other
genome (a dict of thresholds, a port topology, a patch's sizes) works the same way with its
own mutation. Learning inside a life is covered in [learning](settling.md#learning-the-freenudged-rule).

### Any genome

`cadence.genes(space, *, rate=1.0)` returns a mutation for dict genomes over a declared space:

| Kind | Declaration | Mutation |
|---|---|---|
| `log` | `("log", step, low, high)` | multiply by `exp(normal(0, step))`, clip; for scales and thresholds |
| `linear` | `("linear", step, low, high)` | add `normal(0, step)`, clip |
| `int` | `("int", low, high)` | move by one, up or down, within the bounds |
| `choice` | `("choice", option, ...)` | redraw among the options |

Each named gene mutates independently with probability `rate`, which must lie in `(0, 1]`.
Keys that `space` does not name are copied unchanged; a named gene the genome lacks is
skipped. A malformed declaration (unknown kind, wrong length, `step <= 0`, `low > high`, no
options) raises `ValueError`. Any hand-set constant can become a gene; its hand-set value
starts the lineage and is the control.

```python
import numpy as np
from cadence import evolve, genes

space = {"threshold": ("log", 0.3, 0.5, 50.0), "budget": ("int", 1, 20), "law": ("choice", "a", "b")}
start = {"threshold": 4.0, "budget": 3, "law": "a"}

def fitness(genome, seed):  # the caller's: a threshold near 8, a budget near 12, law b
    return -np.log(genome["threshold"] / 8.0) ** 2 - abs(genome["budget"] - 12) / 10 + (genome["law"] == "b")

lineage = evolve(fitness, start, mutate=genes(space), generations=12, population=10, keep=3, seed=5)
assert lineage.best["law"] == "b" and lineage.best["budget"] > 3
assert lineage.best_fitness > fitness(start, 0)  # the hand-set start is the control
```

### Mutation

`cadence.genome.mutate(genome, rng, *, size_step=0.25, fixed=(), tied=())` returns one
`Genome` offspring with the same regions and projections; nothing is added or removed.
A negative or non-finite `size_step` raises `ValueError`, and a name in `fixed` or `tied`
that is not a region raises `KeyError`.

- Each blank region outside `fixed` takes the size `round(size * exp(normal(0, size_step)))`,
  at least one. Designed regions and regions in `fixed` keep their size.
- Each `(leader, follower)` pair in `tied` gives the follower the leader's size, for a
  `prefrontal_cortex` with one neuron per neuron of its region.
- Each projection's `density` is multiplied by `exp(normal(0, 0.2))` within `[0.01, 1]`,
  its `sign` moves by `normal(0, 0.2)` within `[-1, 1]`, and its `scale` is multiplied by
  `exp(normal(0, 0.2))` within `[0.05, 20]`. `count` and `reciprocal` stay as declared.

Fix the regions whose width the environment sets. In `Genome.composed` the motor cortex is
designed and keeps its size, the blank `sensory` region belongs in `fixed`, and the
`prefrontal` region of the working trace follows the `association` region it holds through
`tied`.

```python
import cadence as cd
from cadence.genome import mutate

genome = cd.Genome.composed(2, 2, modules=(12,))
held = (("association", "prefrontal"),)
child = mutate(genome, np.random.default_rng(3), fixed=("sensory",), tied=held)
for parent, offspring in zip(genome.regions, child.regions):
    print(parent.name, parent.size, "->", offspring.size)
```

### Selection

```text
evolve(fitness, genome, *, generations=10, population=8, keep=2, seed=0,
       mapper=map, report=None, mutate=None, grow=None, **mutation)
```

Generation 0 scores the starting genome and `population - 1` offspring of it. Each later
generation scores `population` offspring, each of a parent drawn at random from the `keep`
best of the generation before. Life `index` of generation `g` gets the seed
`seed + 1000 * g + index`; its genome is grown at that seed and scored with
`fitness(grown, seed)`. Scores must be finite, and higher is better.

| Genome | Grown by | Mutated by |
|---|---|---|
| `Genome` | `develop(genome, seed=seed)`, or `grow` | `cadence.genome.mutate`; the `**mutation` keywords go to it |
| anything else | passed as it is, or `grow(genome, seed)` | the `mutate(genome, rng)` you pass (required) |

`evolve` raises `ValueError` when `generations`, `population` or `keep` is not a positive
integer, when `keep` exceeds `population`, when a non-`Genome` genome comes without
`mutate`, and when `**mutation` keywords come with a custom `mutate`.

This fitness lives forty moments, learning which of two actions each cue pays for, and
scores the reward of the last twenty moments less a small cost per synapse. Each grown
anatomy is wrapped with the composed founders.

```python
def life(connectome, seed):
    brain = cd.Brain.compose(2, 2, connectome=connectome, seed=seed)
    rng = np.random.default_rng(seed)
    cue = int(rng.integers(2))
    action = brain.step([np.eye(2)[cue]])
    earned = []
    for moment in range(40):
        reward = float(action[0] == cue)
        earned.append(reward)
        cue = int(rng.integers(2))
        action = brain.step([np.eye(2)[cue]], reward=[reward])
    return float(np.mean(earned[-20:])) - 1e-4 * connectome.synapses

lineage = cd.evolve(life, genome, generations=3, population=4, keep=2, seed=0,
                    fixed=("sensory",), tied=held)
for record in lineage.generations:
    print(record["generation"], round(record["best_fitness"], 3), round(record["mean_fitness"], 3))
```

`Lineage` (`cadence.genome.Lineage`) records what selection did. `generations` holds one
record per generation with `generation`, `best_fitness`, `mean_fitness` and `best`, that
generation's best genome (as its `to_dict()` when it has one, so a `Genome` is recorded as a
dict). `best` is the best genome over all generations, the earliest on a tie, and
`best_fitness` its score. `report=callback` receives the lineage after every generation, so a
long run writes its record as it goes. `Genome.from_dict(record, designed=...)` reads a record
back ([genome records](#genome-records)).

```python
import json
from pathlib import Path

log = Path("lineage.json")
cd.evolve(life, genome, generations=2, population=4, keep=2, seed=0, fixed=("sensory",),
          tied=held, report=lambda so_far: log.write_text(json.dumps(so_far.generations)))
best = cd.Genome.from_dict(json.loads(log.read_text())[-1]["best"], designed={"motor": genome.region("motor")})
connectome = cd.develop(best, seed=0)
```

#### The selection protocol

Select every genome the same way:

- The hand-set genome is the lineage's first member, and the selected genome must beat it on
  held-out worlds, never on the worlds it was selected on.
- A random search over the same space at the same number of evaluations is the second control.
  On a small, smooth space like the one below, random search can match or beat selection;
  report both and claim only what the lineage gains over each.
- The fitness reads only what the genome cannot reweight, and prices both compute and
  surprise. A price on compute alone favours genomes that switch learning off.
- Keep every outcome, including genomes that prune what the designer put in.

```python
space = {"span": ("log", 0.3, 0.5, 8.0), "retain": ("linear", 0.1, 0.0, 1.0), "reads_probe": ("choice", 0, 1)}
hand_set = {"span": 2.0, "retain": 1.0, "reads_probe": 1}

def world(genome, seed):  # a noisy world whose best span is 4 and best retain 0.3
    noise = np.random.default_rng(seed).normal(0.0, 0.05)
    return -np.log(genome["span"] / 4.0) ** 2 - (genome["retain"] - 0.3) ** 2 - 0.1 * genome["reads_probe"] + noise

def draw(rng):  # one uniform draw from the same space
    return {"span": float(np.exp(rng.uniform(np.log(0.5), np.log(8.0)))),
            "retain": float(rng.uniform(0.0, 1.0)), "reads_probe": int(rng.integers(2))}

generations, population = 12, 4
selected = evolve(world, hand_set, mutate=genes(space), generations=generations,
                  population=population, keep=2, seed=0).best
rng = np.random.default_rng(1)
drawn = [draw(rng) for _ in range(generations * population)]   # equal evaluations
lucky = max(enumerate(drawn), key=lambda pair: world(pair[1], pair[0]))[1]
held_out = range(10_000, 10_020)                               # worlds selection never saw
score = {name: np.mean([world(g, s) for s in held_out])
         for name, g in (("hand_set", hand_set), ("selected", selected), ("random", lucky))}
print({name: round(float(value), 3) for name, value in score.items()})
assert score["selected"] > score["hand_set"]
```

### Parallel lives

`mapper` runs the lives of one generation. It receives a callable and an iterable of
`(genome, seed)` jobs, as `map` does, and returns their scores in order. A pool's `map`
gives the same lineage whenever the fitness depends only on its arguments. A thread pool runs
the lives in one process. A process pool (`multiprocessing.Pool.map`,
`concurrent.futures.ProcessPoolExecutor.map`) runs them on separate cores; its workers import
the fitness (and `grow`), so define each as a module-level function in a file and create the
pool under `if __name__ == "__main__":`.

```python
from multiprocessing.pool import ThreadPool

with ThreadPool(4) as pool:
    parallel = cd.evolve(life, genome, generations=3, population=4, keep=2, seed=0,
                         fixed=("sensory",), tied=held, mapper=pool.map)
assert parallel.best == lineage.best and parallel.best_fitness == lineage.best_fitness
```

### What selection, learning and candidate sampling change

| Mechanism | Changes | Leaves as it is |
|---|---|---|
| Selection (`evolve`) | The genome of the next generation: region sizes and projection densities, signs and scales, and so the connectome a life starts from | Every synapse and record inside a life |
| Learning inside a life | Synapses (`Learner`, `ActorCritic`) and memory writes (`SynapticMemory.observe`) | The genome |
| [Candidate sampling](settling.md#detuning-inside-a-life) | The drive of one decision, and so the equilibrium the brain reads | Parameters, records and genome |

A life that samples candidate actions scores the genome together with that sampling. Give
every genome the same perturbation bound, candidate count and settling budget, and count every
candidate's settling steps in the cost of the life.

## Build your own brain

This recipe trains a supervised neural graph on your own data. For a continuing brain, start
with the [world-model lifecycle](start.md#one-continuing-equilibrium-brain) and the
[Brain quickstart](start.md). The blocks run together, in order, on NumPy after you
[install](README.md#install) the package; [orientation](../README.md#for-machine-learning-readers)
explains the terms.

### A supervised graph-learning control

This independent-row classifier tests custom wiring and graph plasticity. A genome names
regions and projections, `develop` creates one connectome, `NeuralGraph` settles it with
reciprocal feedback, and `Learner` updates synapses from free and nudged phases. For
continuing context, memory and recovery use [Brain.compose](tune.md).

#### 1. Encode and wire

A drive has shape `(batch, n)`, one column per neuron; the features, scaled to roughly
unit range, go on the input region's neurons. Labels are integer indices into the
output group.

```python
import numpy as np
import cadence as cd
from cadence.regions import cortex, motor_cortex

FEATURES, CLASSES = 8, 3
rng = np.random.default_rng(0)
centres = rng.standard_normal((CLASSES, FEATURES)) * 1.5

def make(count):
    labels = rng.integers(CLASSES, size=count)
    return centres[labels] + 0.6 * rng.standard_normal((count, FEATURES)), labels

x_train, y_train = make(240)
x_test, y_test = make(120)
```

```python
genome = cd.Genome(
    regions=(cd.Region("features", FEATURES), cortex(24), motor_cortex(CLASSES, lateral=-0.5)),
    projections=(
        cd.Projection("features", "association", reciprocal=False),   # senses drive the cortex one way
        cd.Projection("association", "motor"),                          # reciprocal: the nudge travels back
    ),
)
connectome = cd.develop(genome, seed=0)
brain = cd.NeuralGraph(connectome, cd.learning_neuron_model())
features = list(connectome.populations["features"])
learner = cd.Learner(brain, connectome.populations["motor/actions"], cd.LearnerConfig(eta=1.0, beta=0.1, free_steps=300))

def drive(x):
    d = np.zeros((len(x), connectome.n))
    d[:, features] = x
    return d
```

Hidden neurons learn only where a nudge on the outputs reaches them, so the projection
from the cortex to the motor region is reciprocal. `lateral=-0.5` makes the actions
compete. [Write a cortex](#write-a-cortex) has the catalogue of regions, projections and heads.

#### 2. Calibrate, learn, evaluate

Calibrate on training inputs before scoring held-out data. `calibrate` picks the sampled gain
whose free outputs sit nearest the target, by default a mean highest output activation of 0.5
([calibrating the operating point](settling.md#calibrating-the-operating-point)). This example
declares a small-gain `grid` for its densely connected motor region. `learner.last_calibration`
keeps every candidate and its work; the chosen gain need not make every output responsive.
Each `step` runs a free phase and two nudged phases and moves the synapses once.

```python
learner.calibrate(drive(x_train[:32]), grid=[0.02 * 1.3**k for k in range(16)])
before = learner.accuracy(drive(x_test), y_test)
for epoch in range(3):
    order = rng.permutation(len(x_train))
    for start in range(0, len(x_train), 24):
        rows = order[start:start + 24]
        learner.step(drive(x_train[rows]), y_train[rows])
    print(epoch, "held-out accuracy", round(learner.accuracy(drive(x_test), y_test), 3))
assert learner.accuracy(drive(x_test), y_test) >= 0.95
```

#### 3. Check the settling

The residual checks that a phase satisfies the neuron equations at its state. The
[certificate](settling.md#the-settling-certificate) bounds the distance to the free equilibrium
only while the largest incoming weight mass of any neuron (`row_mass`) is below `mass_limit`
and the neurons have no adaptation; a learned brain often crosses that limit.

```python
free = learner.free(drive(x_test[:8]))
print("largest equation residual per row", learner.brain.residual(drive(x_test[:8]), free).round(4))
cert = cd.certificate(learner.brain)
print("certified", cert.certified, "row mass", round(cert.row_mass, 2), "limit", cert.mass_limit)
```

#### 4. Save and load

```python
learner.save("decision_brain.npz")
again = cd.Learner.load("decision_brain.npz")
assert again.accuracy(drive(x_test), y_test) == learner.accuracy(drive(x_test), y_test)
```

To draw it, record the settlings of a solve and hand them to a page of your own
([the brain in a browser page](tune.md#the-brain-in-a-browser-page)).

#### 5. Grow it

- Reward instead of labels: `ActorCritic` over the same learner ([reward](reward.md));
  `Brain` is the ready composition with a critic, a working trace and associative
  memory ([compose a brain](tune.md#brain)).
- Selection sizes the regions: `evolve` over the genome across lives
  ([evolve a brain](#evolve-a-brain)); any hand-set constant can be a gene.
- A measured connectome: `Connectome.from_synapses`, then a protocol with a shuffled
  control ([the protocol instrument](../benchmarks/instruments/README.md)).
- Speed: `NeuralGraph(..., backend="torch")` for a large blocked connectome
  ([backends](backends.md)).

### Before you trust a number

Score on data the brain never saw, split before any tuning, against a stated control.
Read every `reason` other than an update, and bind seed, version and data to a
[receipt](evaluate.md#receipts). [Evaluate a brain](evaluate.md) has the rules and the
[missteps](evaluate.md#common-missteps).
