# Brains from a connectome

A measured wiring diagram becomes one `NeuralGraph`: every neuron a unit of the
graded rate model, every synapse class an entry with its count and its sign, one
global gain. This page is the recipe, with the numbers that decide each step and
the limits that were measured on wiring at both scales, a 302-neuron nervous
system and a 150,802-neuron brain and nerve cord. The purpose is to import
structural data, supply declared inputs and read model outputs. Biological
dynamics and behavior require their own validation; see the
[numerical and learning contracts](contracts.md). The
[Connectome demo](https://github.com/muellerberndt/cadence-demos/tree/main/connectome)
runs compiled wiring with plastic connections in the browser.

## 1. Custody

Pin the release files by URL and SHA-256, fetch them once, and build a fixture the checks can
rebuild from the same bytes. Record every decision of the dictionary in one place:

- which objects are neurons (the fly: proofread objects of every super class except glia,
  trachea and non-neuronal, 150,802 of 188,508);
- the synapse floor (classes at five or more synapses: 1,877,099 classes carrying 23,131,051
  synapses; the floor decides the sub-net a page can settle later), and the seams the floor
  must not touch: a distributed memory is many weak synapses, and the fly's floor of five kept
  231 of the 1,079 Kenyon cell classes onto one output neuron of the lessons and 14 of the 336
  onto the other, so that what plasticity there could express was measured to be nothing. The
  seam a lesson will move is kept at every count (16,795 Kenyon-cell-to-MBON classes), and
  `seam_report` says what custody left of it before a lesson is designed on it;
- the sign of every class from the presynaptic transmitter (acetylcholine +1; GABA, glutamate
  and histamine -1; modulators 0), which is a declaration, since the predictions are
  per neuron and imperfect;
- what the release lacks (the fly's release holds chemical synapses only, so the giant fibre's
  electrical path to the jump muscles is absent and no fact about it can be asked).

The fixture's manifest (counts, hashes, the dictionary) is committed; the fixture itself is
rebuilt by one command and is not.

## 2. The connectome and its populations

<!-- not-run: reads the fly dataset built in section 1 -->
```python
from cadence import NeuralGraph, Connectome, NeuronModel

connectome = Connectome.from_synapses(n, pre=pre, post=post, count=count, sign=sign, populations=sets)
brain = NeuralGraph(connectome, NeuronModel(gain=0.02), backend="cpu")
```

Name every population the gates and the page will read (`populations` is a dict of name to
neuron indices): the afferents by modality and side, the motor neurons by muscle and side, the
descending neurons by type, the memory site's cells by type. Names are the interface of every
later tool; the fly's `mbon:MBON11:right` is one neuron, `orn:decaying_fruit:left` 130.

The whole brain settles at about 4 ms per step on a laptop CPU at this size; rest is an exact
fixed point of the rectified activation, and activity stayed below one percent of neurons under
every stimulus up to a gain of 0.04. Never call `dense()` on a brain of this size.

## 3. The gain by protocol

One global gain is selected by two or three training facts from the physiology literature and
held out against the rest, with shuffled wirings (the same neurons and counts, postsynaptic
endpoints permuted) each selecting their own gain ([protocols](protocols.md)):

- a fact is a row: a stimulus on a named population at a level, a readout population, a
  predicate; the fly's steering circuit passed 7 of 13 held-out facts at the selected gain
  against 3, 1 and 1 on three shuffled wirings, its instincts 8 of 14 against 2 each;
- readouts over large populations need the `sparse` predicate (a mean over 1,312 descending
  neurons cannot reach 0.5) and left-right sets the `lateralized` one (a steering set holds
  motor neurons with no haltere input, so `active` on the set's mean fails).

Write the facts with their sources before the run, and keep the shuffled wirings' own gains in
the receipt; a control at the measured wiring's gain is a weaker control.

## 4. What a rate model cannot carry

Declare it, supply it, and put the gate on record. The fly's equilibrium reflex at the wingbeat
timescale rests on the spike timing of the haltere afferents; in the closed loop the measured
cord damped roll and yaw and did not damp pitch, so the reflex is the body's own inner loop,
the way the worm's undulation is the body's. Timing-coded pathways, electrical synapses and
neuromodulation are outside the model, and a page that lets the brain override a hand-written
layer keeps the boundary visible.

## 5. One gain, many circuits: a gain per cell class, selected by protocol

At one global gain the fly's antennal lobe ignites. 104 of its 425 local neurons are predicted
cholinergic; they are the broad ones (a median of 88 projection neuron targets each against 16
for the GABAergic ones) and they excite each other through 194,044 synapses, a loop that is
supercritical at the gain the steering circuit selected. Any input lights the lobe into one
state, 38 to 40 percent of the Kenyon cells answer, almost all of them to every odour, and the
codes of the fruit and yeast odours have a cosine of 0.98 at every gain, where the animal's code
is sparse (about 5 percent) and specific. Neuron adaptation did not sparsen it, and neither did a
threshold: with the local neurons' bias at -6 they stayed a third active and the cosine stayed
0.98. A runaway loop is multiplicative and needs a gain.

The remedy is a gain per cell class, selected the way the global gain was:

<!-- not-run: reads the fly dataset built in section 1 -->
```python
def make(attenuation):                       # the candidate: -log gain of the local neurons
    log_gain = np.zeros(C.n); log_gain[C.populations["ln"]] = -attenuation
    return NeuralGraph(C, NeuronModel(gain=GAIN), log_gain=log_gain)

protocol = Protocol(stimuli={"fruit": (...), "yeast": (...)}, rows=[...],
                    training=[("fruit", "kc", "sparse"), ("yeast", "kc", "sparse"),
                              ("fruit", "kc", "specific", "yeast")],
                    levels=Levels(sparse_min=0.02, sparse_max=0.15, specific_max=0.25, code_level=0.05))
attenuation, table = select_gain(make, protocol, [0, 0.7, 1.4, 2.1, 3.0, 4.6], sparsity_cap=0.05)
```

The facts are the animal's (about 5 percent of Kenyon cells answer an odour; the codes of two
odours are specific), the `specific` predicate holds one code apart from another
([protocols](protocols.md)), the candidate is the smallest attenuation that passes them, and
shuffled wirings select their own. Select on the whole brain, as the global gain was: a
sub-net's own stimulated receptors already exceed the sparsity cap. On the fly the local neurons' gain lands at a twentieth of
the measured wiring's: the Kenyon cell codes go to 4 and 11 percent with a cosine of 0.32, 81
cells answer the fruit alone, 372 the yeast alone, 67 both. The number is declared in the
example's dictionary (`CLASS_LOG_GAIN`) next to the global gain, with its receipt.

Measure the code of the site you want to learn at before designing a lesson on it: a code that
is the same for every stimulus carries nothing a lesson can attach to, and no learning rule
repairs that downstream.

## 6. Graph coverage in a browser

The current fly export includes all 150,802 neurons and 1,877,099 connection
classes in its retained fixture. Proofreading and contact-count filters are
importer choices and remain explicit; no additional 60,000-neuron recruitment
cap is used by the current local page.

A reduced graph is an optional approximation requiring its own evidence.
Matching selected population means does not establish equivalent
member-level behavior. Compare individual activities, readouts and task outcomes
under the same stimuli, rather than inferring closure from size alone. Export
in the library's order of synapses (receiving neuron, then sender) so a browser
engine can be held to the library by a parity test.

## 7. Learning on it

The rule is the same actor-critic as on any brain ([learning from reward](reward.md)); the
traps it met on a real wiring are listed there with their readings. Before the first decision,
run the check that reads them all at once:

<!-- not-run: reads the fly dataset built in section 1 -->
```python
report = preflight(brain, outputs, plastic, drives)   # drives: the situations it will decide in
assert report["warnings"] == []                       # each warning names the block that repairs it
```

It finitely settles the brain under the drives and reports a readout with little response (`calibrate_bias`),
a plastic senders' code shared between two situations (a gain per cell class on the `specific`
fact), and a seam too thin for eligibility (custody, `seam_report`); the fly's setup writes
the report into its receipt. On the fly's measured wiring, set up as below, it warns nothing
(the readouts between 0.38 and 0.81, the senders' code 0.23 shared, 75 and 13 active plastic
inputs); on a shuffled wiring the same call warns three times (both readouts pinned at 1.00,
the code 0.82 shared, a thin seam), which is what a control should do. Two decisions come
first:
which neurons read the action (the fly reads approach and avoidance from two mushroom body
output neurons of known valence, one cell each) and which synapses are plastic (the seam the
animal is known to change, `plastic_synapses` on the learner). Then two operating points the
connectome does not carry, both declared and both generic:

- **The readout's threshold.** A connectome says who talks to whom, not how excitable a cell
  is; at one global threshold a readout cell sits on a rail (the fly's approach cell at 1.00
  under every odour, its avoidance cell at 0.01), where the observed nudge has
  little response. `calibrate_bias(brain, drives, {outputs: 0.5}, per_neuron=True)` searches the bias of
  each readout cell by coordinate bisection toward a mean of one half over the
  supplied situations. Coupled cells are revisited, but the finite search can
  miss its target. Check the final full-equation residual and actual means;
  the [qualified calibration option](learning.md#calibrating-the-operating-point)
  refuses an unresolved solve. A target
  such as 0.6/0.4 declares an engineered initial preference; it does not recover
  innate appetite from the connectome. The current export's payload and receipt
  specify the actual calibration. The array goes into `NeuralGraph(bias=...)`.
- **The seam's starting strength.** Synapse counts alone do not identify what
  was inherited versus learned by the specimen. `naive_efficacy(connectome,
  plastic)` gives plastic classes equal effective magnitude as a controlled
  initialization for the learning assay. This changes their measured relative
  strengths and must be reported; it is not a reconstruction of an untrained
  biological brain. The learner starts from it (`NeuralGraph(efficacy=...)`) and the
  measured counts stay in the receipt.

An empty finite `preflight` warning list does not certify equilibrium or useful
acquisition. The selected operating point needs its own numerical check and
free learning assay.

Then the assay: every action the readout can take must have an outcome, and the outcomes the
value can balance; one decision per episode, credited to that decision (the fly's page took a
decision every 0.3 s along an approach and sugar rewarded whichever nudge had been larger).
With the three in place the fly's T-maze reverses in both directions on the library's
actor-critic (sugar at one odour, blows there while the sugar moves, the other found: the
example's `tools/tmaze.py` and its receipt), where before it could not learn one.
A bounded odor assay does not establish reliable food seeking in a
full-scene environment. Transmitter signs are only a simplified fast-interaction map;
receptor-specific modulation, hormone and peptide signaling, innate appetite
and detailed muscle dynamics are not imported by assigning those signs.
