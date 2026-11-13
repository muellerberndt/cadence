# The parrot's brain: two eyes, a map of things, and distance learned

An African Grey sits on a low perch on a table, and things are placed around it. Its brain sees
through two eyes on the sides of its head, finds the things, holds each on a map of the table,
in view and out of it, and learns how far each thing is from its own two eyes and its own head
movements. Nothing tells it what a thing is, where it is or how far.

`cadence.parrot` is imported by name, and `import cadence` loads none of it. This page has two
parts: the first brain as it is built and measured (#332, with the focus of #333), then the
wiring proposal it follows, written from the bird brain's literature, with every source.

| Step | Piece | Section |
| --- | --- | --- |
| A first parrot | `Parrot`, `render`, `Scene`, `Thing` | [A first parrot](#a-first-parrot) |
| What the first brain builds | the regions and the parts that play them | [The first brain](#the-first-brain) |
| How it learns how far things are | the reach and its teacher | [Distance, learned](#distance-learned) |
| What it attends to | `Focus` | [The focus](#the-focus) |
| Which parts are born, and why | the evidence and the learned alternative | [Born parts](#born-parts) |
| Selection over the wiring | `benchmarks/parrot/evolve.py` | [Selection](#selection) |
| The chamber and its readings | `benchmarks/parrot/chamber.py` | [The chamber](#the-chamber) |
| The wiring proposal and its sources | | [The wiring proposal](#the-wiring-proposal) |

## A first parrot

```python
import numpy as np
from cadence.parrot import Scene, Thing, render
from cadence.parrot.brain import Parrot

room = Scene([Thing("apple-1", "apple", 0.05, 0.35, stand=0.1),     # (x, z) on the table, m
              Thing("ball-1", "ball", -0.30, -0.25, stand=0.0)])    # behind the parrot, right
parrot = Parrot(seed=0)
for _ in range(20):
    sight = render(room, parrot.body.eyes())          # the two eyes' images at this pose
    parrot.moment(sight.images)                       # it sees, places, learns, turns its head
assert sight.images.shape == (2, 100, 175, 3)         # left and right, one degree a pixel
assert len(parrot.map.things()) >= 1                  # what it saw is held on its map
assert parrot.focused is None or len(parrot.focused.place) == 2
```

`render` casts each eye's rays over the scene: the table, the room and each thing as a stand-in
of its kind's size and colour (`kinds.json`). `Parrot.moment` takes the two images, taken at
the body's pose, and does one moment of the brain; with `learn=False` it takes no lessons. The
body's eleven joints are `parrot.body` (`Body`: targets in, angles out, a servo of ten moments a
second).

## The first brain

| Bird region | The part | Learned or born |
| --- | --- | --- |
| Retina | The two images, each pixel's colour | Born (the eye) |
| Retina and tectum: figure and ground | `Ground`: the colours that fill a large share of everything seen (and mixtures of two) are ground; a pixel's figure weight is its colour's distance from ground. `components`: the connected pieces of figure | Ground learned from the life; pieces born |
| Rotundus to entopallium: what it is | `Kinds`: category neurons settled on the look (hue, saturation, lightness); a new sort of thing takes a new category | Learned |
| Visual Wulst: two views | Disparity units over the 25 degree overlap (the edges of a thing in the two eyes, each placed inside its pixel by the pixel's colour mixture); parallax units (one eye's sighting before and after the head moved, its shift and how far the eye moved across the line of sight) | Born tuning; the distance each means is learned |
| Arcopallium: the reach | `Distance`: distance neurons log-spaced from 0.12 to 1.2 m, joined to the cues by reciprocal synapses that start at zero | Learned |
| DL: gaze to place | The eye's place (from the head's pose, as the body knows its own head) and the figure's ray, at the distance read | Born geometry of the body |
| Hippocampus: the map | `cadence.space.map.Map` over the table, fed only by the brain's own reports | Carried state |
| Tectum and isthmi: the focus | `Focus`: one winner among the things on the map | Born salience, settled |

A moment, in order: the eyes' figures; each figure seen by both eyes is one sighting; its kind;
its cues (disparity for a thing both eyes see, parallax, size and height in the field for a
thing one eye sees); the distance neurons settle; the place; the map takes the moment's reports;
the teacher's lessons; the focus settles and the head turns.

## Distance, learned

**A thing stays where it is while the head turns.** The eyes sit 1.2 cm in front of the head's
pivot, so a turn moves them. Two sightings of one kind, from two places of the eyes, whose rays
meet (within `agreement_scale`, at an angle of `agreement_parallax` or more) are one thing at
the place where they meet. The distance from each eye to that place is the lesson for the cues
that eye saw, by the library's free and nudged phases. The lessons come from the parrot's own
sightings and its own head; no distance comes from the world.

What the distance neurons learn to read:

- **In front, both eyes**: disparity. Disparity is learned from the head turns, as a cue of one
  moment: at 0.3 m a thing's disparity is 6.9 degrees, and the units place a thing's edges to a
  tenth of a degree.
- **To the sides and behind, one eye**: parallax (the shift of a thing while the head moved),
  and the thing's size and height in the field, which the chamber's stands and scales make tell
  little.
- **Covering one eye** silences the disparity units: in front, distance is then read from one
  eye's cues.

A sighting by one eye on the ray toward a thing of its kind the map holds is kept where the map
holds it (`place_keep`): the distance that keeps it at one place.

## The focus

The focus (#333) is one winner among the things the map holds, in view or out of it. Each
candidate's drive is how much it stands out: it appeared, it moved, it is new to the eyes (the
longer since it was last in focus), it holds (the thing in focus, falling as the parrot
habituates), or it was heard (a cue another part gives). The candidates settle in one layer of
neurons with mutual inhibition, the isthmic nuclei's global inhibition; the most active is the
focus, and the head turns so that it lies in front of the beak, as far as the neck allows.
`Parrot.focused` gives its place, kind and why it won.

## Born parts

The rule: any part a brain can learn is learned. A part kept born states the evidence that birds
are born with it and stays a gene, tried against its learned alternative.

| Born part | Evidence that birds are born with it (sources below) | Its gene, and the learned alternative |
| --- | --- | --- |
| Figure and ground as connected pieces | The tectum's retinotopic map and its competition are laid down before hatching [16, 21]; the ground's colours are learned here | `figure_thickness`; the learned fields of the glimpse (3.2 below) in a later round |
| Disparity and parallax tuning | Disparity-tuned cells of the visual Wulst [30, 32, 33]; the developmental share is open in the sources read | `disparity_form` (edges, energy), `disparity_channels`, `parallax_channels`; binocular fields learned by the fields' rule (`disparity_form: learned`) in a later round |
| Gaze to place | The step uses the eye's pixel layout and the head's pose: the body's own anatomy, as the tectum is born retinotopic [16, 21] | `place_step: geometry`; the learned gain field (round 2) |
| The focus's salience | Born salience in the tectum and isthmi [14, 15, 20] | `focus_*`, `novelty`, `isthmi_*`; salience learned from rewarded looks (#333) |
| The teacher's agreement | The comparison of two views of one thing, as birds that move their heads to judge depth [53, 60, 61, 62] | `agreement_*`; the beak's reach (touch) as the other teacher |

## Selection

The genome is one dict of named genes (`cadence.parrot.brain.GENES`); names follow the gene
space below. `benchmarks/parrot/evolve.py` selects with `cadence.evolve` and `cadence.genes`
over 25 of them, each genome living worlds its generation shares, scored by the map's error in
view and out of view, identity, distance and compute, each against the hand-set brain. Random
wiring of the same size (`wiring: "random"`) and random search at equal evaluations are its
controls.

## The chamber

`python -m benchmarks.parrot.chamber --founders 900 --out runs/dev` lives one founder and reads
E1 (seen and held), E2 (one thing from another view), E3 (distance learned, with an eye covered),
E7 (the probe's panel, the distance neurons found, silenced, and as many others silenced), and
the panel of things on the table. The declared protocol is
[benchmarks/parrot/PROTOCOL.md](../benchmarks/parrot/PROTOCOL.md). One moment costs 23 ms on
one core of the chamber's machine: rendering both eyes 9, early vision 6, the map 4.

## The wiring proposal

The rest of this page is the wiring proposal the first brain follows and the gene space
selection searches, written from the bird brain's literature: which regions of a bird's brain
it follows, which Cadence part plays each, how distance is learned from two eyes and the head's
own movements, and which wiring choices are genes. The first brain builds the parts named above;
the others (the association with its working trace, the reach and next-view slots learned by
reward, the glimpse's learned fields, hearing and the voice) join in the steps that follow.

## Summary

```text
  LEFT EYE (175 deg)                                 RIGHT EYE (175 deg)        11 JOINTS
        |  \                                        /  |                           |
        |   +------- 25 deg overlap in front ------+   |                    proprioception,
        |     (each eye feeds the opposite tectum)     |                    head direction,
        v                                              v                    efference copy
  RIGHT TECTUM <------- isthmi: local feedback, -----> LEFT TECTUM                |
  priority map          global inhibition              priority map               |
  and winner                                           and winner                 |
        |  \_______ orienting reflex: up, down, left, right _______> head slots   |
        |  gated glimpse at each winner                                           |
        v                                                                         |
  ROTUNDUS -> ENTOPALLIUM (what)            GLd -> WULST (how far) <--------------+
  form, colour, motion, looming;            two views of one thing: disparity in the
  kinds = learned categories                overlap, parallax before and after a move
        |                                          |
        v                                          v
  ============ NIDOPALLIUM / NCL: the association and its working trace ============
        |                  ^                       |                    ^
        v                  | where, what           v                    | dopamine
  DL: place =              |              ARCOPALLIUM: motor slots      |
  eye + distance x ray     |              head yaw, pitch, roll; neck;  STRIATUM, VTA, SNc:
        |                  |              beak; wings; grips; legs;     actor and critic.
        v                  |              reach (the distance);         The pistachio pays;
  HIPPOCAMPUS: the map ----+              next view (a prediction)      views that agree pay
  of things, held out of view
```

1. The wiring follows the bird's two visual pathways: the tectum finds things and turns the
   head to them (where), the rotundus and entopallium learn what they are, and the Wulst compares
   two views of one thing (how far).
2. How far a thing is: the one distance at which two views of it agree. The two views are the
   two eyes at once, in the 25 degree overlap in front (a 3.6 cm baseline), and one eye before
   and after the head moves, at the sides (up to 4.6 cm when the neck leans).
3. Two teachers, neither of which tells a distance. Sight teaches the brain to predict where a
   thing appears after its own head movement. The map pays when the places the brain writes for
   one thing from two views agree.
4. The hippocampal formation is the place module (`Map`), fed only by the brain's own reports:
   its learned kinds, and places from its own gaze and its own distance.
5. The caudolateral nidopallium (NCL) is the association region with its working trace. The
   arcopallium is the motor: one slot per joint, plus a reach slot for each thing looked at,
   which says its distance.
6. Dopamine is the actor-critic's prediction error. The pistachio pays, and views that agree pay.
7. Library parts used: learned receptive fields, `PopulationCode`, `Map`, `Brain.compose` with
   `slots`, the efference copy, the working trace, the actor-critic and arousal, `evolve` and
   `genes`. The learned categories come from the retina lane (#330).
8. New parts:
   - the tectal priority sheet with its winners;
   - the glimpse the winner gates;
   - the two-view units;
   - the gaze-to-place step;
   - the agreement reward;
   - a head-bob rhythm, held at zero.
9. The GA searches 67 named genes plus the density, sign and scale of every projection. The genes
   cover sizes, which projections exist, receptive fields, rates, delays, where disparity is
   computed, and whether the head bobs. Every gene starts at its hand-set value, and that value
   is the control. The first lineage searches the 22 genes marked F.
10. Fitness is the map's error in view and out of view, plus identity, distance error and
    surprise, less compute. It is read against three controls: the hand-set wiring, random wiring
    of the same size, and random search at equal evaluations. Every reading uses twelve fresh
    founders.

Numbers in brackets refer to the [sources](#sources) at the end.

## 1. What the parrot needs, first and later

| First (#332) | Later (#331) |
| --- | --- |
| Two lateral eyes: a wide field each, a narrow overlap in front, a blind spot behind | Hearing: questions asked aloud |
| Finding things in the images, telling them apart, turning the head to them | The voice: words said letter by letter, the parrot's core and shell vocal system |
| Holding things on a map of the table, in view and out of view | Words bound to things seen and heard |
| Distance, learned from two eyes and from its own head movements | Reward for right answers (the pistachio) through the basal ganglia |
| Head and neck, legs, beak and wings wired to the brain | A sense of the present moment against the past (the map's clock, records of events) |

The parrot of the chamber has the following body and eyes, and the numbers in this page are
computed for it.

- **Eyes.** Each eye image is 175 by 100 pixels at 1 degree a pixel. Each eye sees 175 degrees.
  The two fields overlap 25 degrees in front of the beak and leave 35 degrees unseen behind the
  head. The eyes are fixed in the head, 3.6 cm apart and 1.2 cm in front of the head's pivot.
  The image has no depth channel.
- **Joints.** There are 11: head yaw, pitch and roll, neck pitch, beak, two wings, two grips and
  two legs.
- **Rate.** Ten moments a second.
- **Table.** 1.6 by 1.1 m. Things stand on it, or on clear stands up to 0.25 m high, at
  scales of 0.7 to 1.4.

## 2. The bird brain, region by region

### 2.1 The parrot's eyes and fields of view

- **Senegal parrot fields** [56]. The Senegal parrot is the nearest parrot whose visual field is
  measured.
  - Each eye's field is 186 degrees in the horizontal plane.
  - The binocular field is at most 27 degrees wide, at the horizontal, and about 20 degrees
    through most of its vertical extent. It narrows to 5 degrees where the bill tip projects.
  - The blind sector behind the head is 16 degrees wide at the horizontal.
  - The eyes move up to 24 degrees.
  - The bill tip lies at the edge of the field, with a blind area beneath it. To see things
    below the bill, the parrot pitches its head forward to view them with both eyes, or turns
    its head to view them with one eye. Parrots also bring objects up into view with their feet.
- **Parrot retina** [57]. The budgerigar and Bourke's parrot have an area centralis without a
  fovea.
  - Peak ganglion cell densities are 22,300 to 34,200 and 18,100 to 38,000 cells per mm².
  - Estimated acuity is 6.9 and 9.2 cycles a degree. The chamber's eye, at 1 degree a pixel,
    resolves 0.5 cycles a degree, more than ten times coarser.
- **Binocular fields across birds.** Most birds have binocular fields of 20 to 30 degrees, and
  the widest measured is 61 degrees [49, 50, 52].
  - Except possibly in owls, the binocular field is a by-product of each eye seeing across the
    midline. It gives the direction of travel and the time to contact about the bill [51, 52].
  - The high-resolution regions of the retina look sideways [50].
  - Birds that move their eyes often abolish the binocular field, and one eye controls pecking as
    accurately as two [49].
- **The chamber's eyes against the Senegal parrot.** The chamber's eyes (25 degree overlap, 35
  degree blind sector) sit close to these numbers.

### 2.2 The tectofugal pathway: retina, optic tectum, nucleus rotundus, entopallium

- **Retina to tectum.** The optic tectum receives more than 90% of the retina's projections
  [11].
  - In the pigeon each tectum receives only the opposite eye, topographically, at up to 15,000
    labelled cells per mm². The thalamofugal relay receives fewer than 150 [21].
  - The tectum is laminated, with retinotopic input and separate input and output layers. Its
    role is orientation and the spatial analysis of what the senses report [16].
- **A map of priority.** The tectum has superficial layers 2 to 9 and intermediate and deep
  layers 10 to 13. Its neurons encode a stimulus's strength, novelty and relevance as firing
  rate in a spatial map, with little selectivity for colour, orientation or direction [14].
  - About 30% of owl tectal neurons fall silent, switch-like, when a competitor 30 degrees away
    grows stronger than the stimulus in their field [14].
- **The isthmic nuclei pick a winner and gate what goes up** [14, 15, 20].
  - Nucleus isthmi pars parvocellularis (Ipc) and the semilunar nucleus (SLu) return cholinergic
    feedback to the tectal spot that drives them.
  - Pars magnocellularis (Imc) inhibits broadly, so the strongest stimulus wins.
  - In the pigeon, Ipc receptive fields are 15 to 25 degrees across. A second moving spot
    suppresses Ipc firing at the first site after 35 to 100 ms, even 100 degrees away.
  - The upper and central fields suppress the lower field more than the reverse [20].
  - Blocking Ipc removes the responses of nucleus rotundus to targets at that place, so the
    isthmi gate which place the tectum sends up to the thalamus [20].
  - Rotundus neurons, with large receptive fields, relay the location of highest priority to the
    entopallium [15].
- **Top-down gain.** The arcopallial gaze field raises tectal gain at a location. It biases the
  competition but does not override a much more salient stimulus [15, 44].
- **Nucleus rotundus loses retinotopy and splits by feature** [12, 13].
  - Anterior rotundus responds to colour and luminance, central and posterior rotundus to motion.
  - Some rotundus neurons signal time to collision of an approaching object, in three types
    (tau, rho and eta) [27, 28].
  - Tectal neurons that fire for a spot moving against its background separate the motion of
    things from the bird's own motion [26, 40].
- **Entopallium and its outflow** [12, 23, 24, 25].
  - The entopallium keeps rotundus's split. Anterior lesions impair colour and form tasks,
    posterior lesions motion tasks.
  - Through its belt it projects to the frontolateral (NFL), intermediate-lateral (NIL) and
    caudolateral (NCL) nidopallium. It is reciprocally connected with the mesopallium, and its
    columns continue into the lateral striatum.
  - The visual dorsal ventricular ridge is three layers joined by reciprocal, topographic
    columns: entopallium, intermediate nidopallium (NI) and ventral mesopallium. Its outputs to
    the rest of the pallium leave from NI only. NI is reciprocally connected with the Wulst and
    the lateral striatum [25].
  - Such columns repeat across the bird's sensory pallium [6].
- **The pathway's role in lateral-eyed birds.**
  - It both locates and identifies. In a proposed loop, the tectum locates and guides fixation,
    the higher stations analyse the fixated object, and the result returns to the tectum to
    decide what to do [19].
  - In the pigeon the tectofugal pathway serves the frontal field and the thalamofugal pathway
    the lateral field. Lesions of the two give a double dissociation [22].
  - Parrots have a relatively small tectum, rotundus and entopallium [29] inside a relatively
    large telencephalon [9].
- **Turning the head** [42, 43, 44, 98].
  - Downstream of the tectum's map, four separate circuits drive upward, downward, leftward and
    rightward head movement [42]. Stimulating the tegmentum gives fixed-direction head saccades
    in six directions [43].
  - A crossed projection from the deep tectum reaches the head-neck premotor neurons of the
    reticular formation [98].
  - The anterior arcopallium holds a gaze field that projects in parallel to the deep tectum and
    to the saccade generators [44].

### 2.3 The thalamofugal pathway: dorsolateral geniculate, visual Wulst, and disparity

- **The route.** Retina, dorsolateral geniculate (GLd, the nucleus opticus principalis
  thalami), then the visual Wulst. The Wulst is the homologue of primary visual cortex
  [11, 17], and its roles are spatial orientation, motion and binocular vision [11]. The two
  pathways run in parallel to separate telencephalic targets [18].
- **Disparity in owls.**
  - Owl Wulst neurons are selective for orientation, direction and binocular disparity [30].
  - Output neurons demand a specific horizontal disparity over receptive fields many times
    larger than the optimal stimulus [31].
  - In awake owls viewing random-dot stereograms, 76% of 160 Wulst neurons were tuned to
    horizontal disparity, with Gabor-like curves that fit the disparity-energy model and no
    disparity columns [32]. Neurons with longer latencies resolve the false matches of shorter
    ones, which suggests a hierarchy [33].
  - Owls have about 50 degrees of binocular overlap, and about 85% of their Wulst cells are
    binocular and disparity-sensitive [11].
- **Lateral-eyed birds.**
  - In the zebra finch the Wulst maps the opposite field from 5 degrees past the beak to 125
    degrees sideways, with no over-representation of the fovea.
  - Input from the eye on the same side reaches the zebra finch Wulst by recrossing thalamic
    fibres, and it inhibits.
  - The zebra finch authors propose that binocular processing in the Wulst is a specialty of
    birds with frontal eyes [34]. The zebra finch Wulst holds three retinotopic maps [35].
  - Wulst size grows with the width of the binocular field [36]. In the pigeon the thalamofugal
    pathway has a very limited role in binocular integration [21].
- **Depth with two eyes, behaviourally.** Pigeons discriminate a near target displaced in depth,
  covering one eye disrupts the task, and the cue is relative disparity [63]. Their stereoacuity
  is 0.8 to 1.8 minutes of arc [64].
- **Pigeon and dove neurons.** No disparity neurons are found in doves. The pigeon's relative
  depth is attributed to convergence and accommodation [50].
- **Other species.** Owls use relative disparity and segregate figure from ground by disparity
  alone [65]. A report of stereopsis in a falcon [66] is not replicated [50].
- **The Wulst's outputs.**
  - Its only efferents leave from the hyperpallium apicale [37].
  - The rostral Wulst sends a tract to the cervical spinal cord, the red nucleus and the pontine
    nuclei [38].
  - The hyperpallium apicale projects to the optic-flow nuclei LM and nBOR [41].
  - The rostral Wulst is somatosensory [11, 17].

### 2.4 How a bird with lateral eyes judges distance

| Cue | Evidence in birds | In the chamber's parrot |
| --- | --- | --- |
| Binocular disparity in the frontal overlap | Owl Wulst [30, 32, 33, 65]; pigeons near and relative [63, 64] | 3.6 cm baseline: 20.4 degrees at 0.1 m, 6.9 at 0.3 m, 2.1 at 1.0 m |
| Motion parallax from the head's own translation | Pigeons stop the head twice before a peck [60]; simulated parallax moves the head's stop [61]; head movements serve depth [53]; owls peer before attacking [62] | A full neck lean moves the eyes 4.6 cm forward: a thing straight to the side shifts 8.7 degrees at 0.3 m and 2.6 at 1.0 m. A crouch moves them 2.5 cm down |
| Head-bobbing | The head holds steady in space, then thrusts. Bobbing is visually driven [58], locked in the hold [59] and timed with the legs [54]. Its proposed roles are biomechanics, depth from parallax and image stability, and shape is seen during the thrust as well [68] | A head-bob rhythm is a gene, held at zero (section 4) |
| Expansion and time to contact | Rotundus looming cells [27, 28]; the flow about the bill gives time to contact [51] | A lean toward a thing in front expands its image |
| Accommodation and convergence | Corneal and lenticular accommodation, up to about 9 dioptres corneal in the pigeon [55, 67]; convergence in pigeons [50] | Absent: the image has no blur and the eyes are fixed |
| Height in the field, familiar size | A geometric cue; no bird source read | Removed by the chamber's random stands and scales |

The geometry sets a floor on the error a cue can reach.

- **The rule.** A thing at distance d seen over a baseline b shifts by about b/d radians. The
  relative error of a distance read from a shift misjudged by δ radians is d·δ/b.
- **At the chamber's sizes.** At a quarter-pixel precision (0.25 degree):

| Distance | Stereo error (3.6 cm) | Lean error (4.6 cm) |
| --- | --- | --- |
| 0.3 m | 3.6% | 2.9% |
| 0.5 m | 6.1% | 4.8% |
| 1.0 m | 12.1% | 9.5% |

- **Where each cue works.** In front, the lean moves the eyes along the line of sight, so a thing
  ahead does not shift and only grows. In front, then, two eyes carry the distance. At the sides,
  the lean carries it.
- **What this predicts.**
  - Covering one eye costs distance in the front overlap and little at the sides (#332 E3).
  - A brain with one eye covered can recover some frontal distance from the vertical parallax of
    a crouch: 4.8 degrees at 0.3 m for 2.5 cm. E3 is therefore read together with how much the
    brain moved its head.

### 2.5 Self-motion, gaze and a stable head

- **Optic-flow nuclei.** The nucleus of the basal optic root (nBOR) and the pretectal nucleus
  lentiformis mesencephali (LM) have large receptive fields, about 60 degrees, tuned to slow
  whole-field motion. They feed the vestibulocerebellum: the flocculus codes self-rotation and the
  uvula and nodulus self-translation [39].
- **Object motion and self-motion.** Tectofugal motion cells read local motion and are inhibited
  by whole-field motion. Some nBOR cells are binocular and tuned to translation or rotation [40].
- **A stable head.** With the head free, the head supplies about 80% of vestibular gaze
  stabilisation [47]. Pigeons stabilise with the head, quail with head and eye together [48].
  The pigeon's head is held almost perfectly against rotation and largely follows translation
  [46].
- **Head as gaze.** In pigeons the head is a proxy for gaze, and head saccades scan the lateral
  field [45].

### 2.6 How the pallium is organised

- **Names and homologies.** Most of the bird telencephalon is pallium comparable to the
  neocortex. The old names hyperstriatum, neostriatum and archistriatum are hyperpallium,
  nidopallium and arcopallium [2]. The rationale for the new names is in [1].
- **Columns.**
  - Functional columns run from primary sensory input in the intercalated pallium to the
    nidopallium and hyperpallium, then the mesopallium, then the arcopallium as the output [5].
  - A cortex-like canonical circuit repeats in columns linked sideways to associative and motor
    areas [6].
- **Hubs and neurons.**
  - The pigeon's forebrain network has five connector hubs: dorsal and intermediate arcopallium
    (AD, AI), NCL, the dorsolateral corticoid area (CDL) and the area parahippocampalis (APH) [7].
  - Parrots and corvids have primate-like numbers of forebrain neurons, with a high share in
    the pallium [8].
  - Parrots have an enlarged medial spiriform nucleus. It relays the telencephalon to the
    cerebellum, and the return paths reach the rostral Wulst and NCL [10].
- **Cognition.** Birds reach primate-like cognition with this pallium [3, 4].

### 2.7 The hippocampal formation and spatial memory

- **Size and memory.** Food-storing birds have a larger hippocampus, and hippocampal damage
  disrupts memory for cache sites [69, 73]. Jays remember what they cached, where and when [74].
  Spatial learning is not limited to specialist species [72], and the hippocampus is needed to
  learn map-like representations for homing [71].
- **Anatomy and connections** [70, 75].
  - The pigeon's hippocampal formation has seven subdivisions.
  - The dorsolateral part (DL) takes its main input from the hyperpallium densocellulare of the
    Wulst, reciprocally.
  - The dorsomedial part (DM) is the exit to septum and hypothalamus, and it is reciprocally
    connected with the arcopallium.
- **Place cells and replay.** In the tufted titmouse, place cells concentrate in the anterior
  hippocampus. They rise from under 10% of excitatory cells at the posterior pole to over 70% at
  the anterior pole, and sharp-wave ripples are found [76].
- **A barcode per cache.** In chickadees each caching event writes a sparse "barcode" unique to
  that cache, beside the smooth place code. The barcode returns when the cache is retrieved [77].
- **An entorhinal-like gateway** [78].
  - In chickadees, DL with CDL is bidirectionally connected with the hippocampus. It carries
    border cells, grid-like cells, head-direction cells, speed cells and conjunctive cells.
  - Its pallial inputs come from the hyperpallium densocellulare and from NFL and NIL, which are
    likely tectofugal targets.
- **Direction and place in other species.**
  - About 12% of quail hippocampal neurons are head-direction cells. Their tuning is broad and
    rotates with a salient visual cue [79].
  - Owls flying between perches carry place, flight-direction and perch-position cells in the
    anterior hippocampus and the posterior hyperpallium apicale [80].

### 2.8 The caudolateral nidopallium (NCL), the prefrontal analogue

- **The analogue.** NCL corresponds to the prefrontal cortex on anatomical, neurochemical,
  physiological and behavioural grounds [81].
- **Its connections** [83].
  - It is reciprocally connected with the secondary sensory areas of every modality. It also
    receives from the deep Wulst, NFL and the thalamus.
  - It projects to the basal ganglia, mainly to the somatomotor intermediate arcopallium, and to
    the amygdala-like posterior and ventral arcopallium.
- **Dopamine.** Of the forebrain, NCL holds the densest dopamine innervation. It comes from the
  ventral tegmental area and substantia nigra and forms basket-like fibres [84, 82]. In crows it
  splits into several subareas [85].
- **Working memory and rules.**
  - NCL neurons hold delay activity that relates to success [86].
  - 67% of NCL cells are delay cells, and in 88% of cases their sustained activity follows a
    "remember" cue and stops after a "forget" cue [87].
  - In crows, NCL neurons hold the current abstract rule [88] and the stimulus held in working
    memory [89]. The working-memory capacity limit arises as in monkey prefrontal cortex, by
    divisive normalisation [90].

### 2.9 The arcopallium and the motor output

- **Divisions** [91, 92, 93].
  - The arcopallium is a major source of descending projections.
  - The intermediate (AI) and dorsal (AD) parts project to thalamus, tectum, tegmentum,
    reticular formation and brainstem.
  - The medial, posterior and ventral parts are amygdala-like.
  - Each sense keeps a loop of its own: the visual part of AI sends a massive projection to the
    deepest tectal layer [94].
- **Beak and tongue.** The beak and tongue loop runs from nucleus basalis to the frontal
  nidopallium, then to lateral AI, then to the pontine and medullary reticular formation and the
  medial spiriform nucleus [95].
- **Reticular premotor zones** [96, 97].
  - The bulbar reticular formation holds premotor zones for jaw and neck, jaw, tongue and neck,
    and tongue and locomotion [96].
  - Head-neck premotor neurons lie interleaved with those of jaw and tongue, which suits feeding
    [97].
- **Head and beak as limbs.** Climbing parrots use the beak and neck as a third limb [99].
- **Two further descending paths.** The rostral Wulst has a descending "pyramidal" tract [38].
  The cerebellar loop through the medial spiriform nucleus is enlarged in parrots [10].
- **Wings and legs.** No source read here describes the pallial control of wings and legs.

### 2.10 The basal ganglia and dopamine: the pistachio

- **Anatomy** [2, 100].
  - The striatum is medial (MSt) and lateral (LSt), and the pallidum is single. The substantia
    nigra has pars compacta and reticulata, and the rostroventral striatum is a nucleus
    accumbens.
  - The ventral tegmental area supplies dopamine to medial MSt. The substantia nigra pars
    compacta supplies lateral MSt and LSt.
  - Birds have direct and indirect pathways, and a limbic loop through accumbens, ventral
    pallidum and thalamus.
- **Reward in the chick striatum.**
  - Lesions of the chick ventromedial striatum impair learning which colour predicts reward
    [100], and lesions of its caudal part make chicks choose the small immediate food [110].
  - Its neurons fire to the reward and to cues that predict it [109], coding how soon and how
    much [111].
- **What dopamine neurons signal in songbirds.**
  - They signal performance error: firing drops about 58 ms after a worse-than-expected syllable
    and rises when an expected error is omitted. They do not respond to the same sounds played
    back [103].
  - Cues that predict water drive dopamine in both the vocal and a medial striatal target.
    Errors of the song reach the vocal target only [104].
  - Dopamine neurons are disinhibited through a basal ganglia loop [105].
- **What dopamine does in the striatum.**
  - Long-term potentiation in the vocal striatum needs D1 and NMDA receptors [106].
  - Dopamine to the vocal striatum is sufficient to steer what the bird learns [107]. One
    circuit serves both copying a tutor and learning from external reward [108].
- **Reviews.** A reinforcement-learning account of the songbird loop [101] and a review [102].
  Outside the basal ganglia, D1 receptors outnumber D2 [82].

### 2.11 Later: hearing

- **The pathway.**
  - The cochlear nuclei: nucleus magnocellularis (NM) and nucleus angularis (NA).
  - NM feeds nucleus laminaris (NL). NM's axons from both ears are delay lines, and NL's neurons
    detect when they coincide, which maps the time difference between the ears [114, 116].
  - NA and NL project to the midbrain nucleus MLd [117], and on through the thalamic nucleus
    ovoidalis to Field L2.
  - From Field L2 to L1 and L3, and on to the caudomedial nidopallium and the caudal mesopallium
    [118, 120, 121].
- **What the auditory areas hold.** Natural sound statistics shape their tuning, and the
  secondary areas are plastic [112]. The memory of a tutor's song lies in the auditory
  association regions [115].
- **Auditory space.** It is coded in parallel channels [113]. The tectum's visual map teaches
  the auditory space map where sounds are [122, 123].
- **Parrots.**
  - In the budgerigar, the thalamic auditory relay projects to Field L and the adjacent NIDL.
    NIDL also receives a visual thalamic input, which suggests a place where sounds and sights
    meet [119].
  - Budgerigars hear from 77 Hz to 7.6 kHz, best at 3 kHz [124].

### 2.12 Later: the parrot's vocal system, its core and its shell

- **Two pathways.** Vocal learners have a posterior production pathway and an anterior pallial,
  basal ganglia and thalamic loop. Vocal learning evolved by duplicating a motor pathway [125,
  126].
- **A song system within a song system** [127].
  - The parrot's "core" song system resembles that of songbirds and hummingbirds. The "shell"
    song system around it is unique to parrots.
  - The African grey is among the species with the smallest cores and correspondingly larger
    shells.
  - Cores connect mainly to cores, and shells to shells.
  - Only the core of the central nucleus of the anterior arcopallium (AAC) projects to the
    brainstem vocal motor neurons. The AAC shell projects to adjacent forebrain song nuclei.
  - In a preliminary lesion study, lesions of the core of the central nucleus of the lateral
    nidopallium (NLC) shifted vocal frequency fast. Lesions of the NLC shell slowly shrank the
    contact-call repertoire.
  - The shells have no striatal or thalamic counterpart, so core and shell may share one
    striatum and thalamus [126].
- **Production and the anterior loop in the budgerigar** [128, 129, 131].
  - Production runs from NLC to AAC, then to the midbrain nucleus DM and the hypoglossal
    motor nucleus that drives the syrinx.
  - The anterior loop runs from the oval nuclei of the anterior mesopallium and nidopallium (MO,
    NAO) to the magnocellular nucleus of the medial striatum (MMSt), then to the thalamic nucleus
    DMm, and back to NAO.
- **Parrot circuits are their own** [130, 132].
  - Auditory input to the budgerigar vocal system comes from nucleus basalis and the frontal
    nidopallium, and its forebrain circuits evolved apart from those of songbirds [130].
  - Lesions along the pathways tied to nucleus basalis abolish calls. Lesions of Field L2a
    spare them [132].
- **A vocal motor map** [133]. In the budgerigar's AAC core, 51% of 220 neurons are tuned to
  pitch, tiling about 1 to 5 kHz, and their activity leads each vocal element.

### 2.13 Where the sources stop

- **No African grey numbers.** No source read measures the African grey's visual field, retina
  or Wulst. The nearest are the Senegal parrot [56] and the budgerigar [57].
- **Disparity in lateral-eyed birds.** Where, or whether, a lateral-eyed bird computes disparity
  is open. In owls it is computed in the Wulst [30, 32]. In pigeons depth is discriminated with
  two eyes, but no disparity neurons are found [50, 63], and the left rotundus integrates both
  eyes [22].
- **Head movements in parrots.** No study read shows parrots using head movements to judge
  distance. The evidence is from pigeons and owls [60, 61, 62].
- **Timings and projections.** No numeric head-bob timing was found in the texts read. A
  projection from the Wulst to the tectum was not found in what was read.
- **Dopamine and vocal anatomy.** Dopamine physiology comes from songbirds, chicks and pigeons;
  none was found for parrots. A projection from the auditory LAN to NLC in parrots was not
  verified.
- **How the gaps are handled.** Where the sources stop, the wiring below makes a choice. That
  choice is a gene, so that selection decides.

## 3. The parrot's first brain in Cadence parts

### 3.1 The rules the wiring keeps

- **Everything the brain knows is learned:** what a thing is, where it is and how far it is.
  Nothing in the brain's path reads the chamber's per-pixel truth.
- **A designed part is a gene, with its hand-set value as the control, or it is a control.**
  The retina's arithmetic, the tectal competition, the step from gaze to place, the agreement
  reward and the head-bob rhythm are designed. Each has its genes in section 4.
- **The body's own geometry may be designed; nothing else that sets a place may be.** The step
  from gaze to place uses the eye's pixel layout and the head's pose from proprioception. Those
  are the body's anatomy, the way a bird's tectum is born retinotopic [16, 21]. The thing's
  place comes from the brain's own finding and the brain's own distance.
- **Every answer is a settled state.** The tectum's winner, the categories, the association and
  the map each settle by the library's neuron rule. A feed-forward readout or a lookup is a
  control.

### 3.2 Regions

Sizes are the hand-set founder. The range a gene allows is in section 4.

| Bird region | Cadence part | Size | Learns, and from what | In the library |
| --- | --- | --- | --- | --- |
| Retina | Fixed arithmetic on each eye image: the opponent channels of the learned fields' retina (luminance, red against green, blue against yellow), a change channel (each patch against the moment before), pooled more finely about the eye's axis | About 1,100 receptors a channel per eye | Nothing | Opponent retina: `LearnedFields`. Foveated pooling in two dimensions: new (the retina lane's `Eye` spacing, in one dimension) |
| Optic tectum with the isthmic nuclei, one per eye, each fed by the opposite eye [21] | A designed sheet. Salience is contrast, colour contrast and change. Local feedback (Ipc, SLu) and global inhibition (Imc) settle to k winners. The Eyes demo's collicular map and gaze slots are the working reference | 44 by 25 cells each, 4 degrees a cell; 1 winner each | Salience is fixed. A top-down projection from the association (the arcopallial gaze field) learns from the actor's reward | New: a designed `Region(circuit=...)` |
| Nucleus rotundus to entopallium (what) | `learned_fields` read on a glimpse: a window about each winner, the window the isthmi gate [20]. Adds 8 motion units (opposed to the whole field's flow [26]) and 3 looming units [27, 28] | 32 fields by 4 pools, plus 11: 139 a glimpse | The fields learn by their Hebbian rule on the residual, at every accepted moment | Fields: exist. The glimpse window, motion and looming units: new |
| Entopallium to NI, kinds | The learned categories of the retina lane (#330), on the glimpse's form and tint | 24 categories | A category holds the mean of what it was seen as. A new sort of thing takes a free category | Retina lane (#330) |
| Dorsolateral geniculate to visual Wulst (two views) | Binocular units at the winner when it lies in the 25 degree overlap, tuned to 12 disparities (energy-model units [32]). Parallax units per eye, tuned to 12 shifts of the winner between a moment and the moment k before | 12 + 2 x 12 = 36 | Hand-set tuning. In the learned arm, binocular fields learn by the fields' rule (a gene) | New |
| Optic-flow nuclei (nBOR, LM) | The mean flow of each eye's whole field, in 4 directions | 2 x 4 | Nothing | New (small) |
| Proprioception; head-direction and speed cells [78, 79] | A `PopulationCode` per joint angle (8 units) and velocity (4 units), and the head's yaw as a ring of 16 head-direction units | 148 | The code is fixed; the synapses that read it learn | `PopulationCode` |
| Efference copy | What the brain just commanded | One fading value per motor neuron | The synapses that read it learn | `efference_amplitude` |
| DL and CDL, entorhinal-like [78] | The step from gaze to place: place = eye + distance x ray, projected on the table plane | (arithmetic) | Nothing in round 1. The learned arm (round 2) is a gain-field region that learns from the agreement of views | New, in the parrot's own code |
| Hippocampus | The place module `Map` over the table: 1.6 by 1.1 m, 32 by 22 cells (5 cm), units for 24 kinds. It is fed by the brain's own reports: kind = its category, place = the gaze-to-place step at its own distance. `lit` = the cells inside the glimpse windows, out to the distance read (section 3.6) | 704 cells | Nothing: carried activity, with no taught synapse | `cadence.space.map.Map` |
| Nidopallium, mesopallium and NCL (the association hub [7, 83]) | The composed brain's `association` | 256 | Lessons of the next view, and rewards through the actor-critic | `Brain.compose` |
| NCL delay cells [86, 87] | The working trace (`prefrontal`); its carried gain stands for the remember-or-forget gate | 256 | The synapses that read it learn | Exists |
| Arcopallium: AI, AD, the gaze field [44, 91] | Motor slots:<br>- head yaw 7, head pitch 7, head roll 3, neck pitch 3, beak 3;<br>- wings, grips and legs 3 each (left and right);<br>- reach 12 per winner (the distance, log-spaced 0.08 to 1.1 m);<br>- next view 9 per winner (the shift expected)<br>Two winners at the hand-set | 83 | Joints and reach: the actor-critic. Next view: lessons from the next moment's sight | `slots` |
| Tectum to the reticular formation to the neck [42, 43, 98] | A designed one-way projection. Each tectum's winner column drives the head yaw and pitch slots through four directional channels (up, down, left, right) | 4 channels | Fixed; its gain is a gene | New wiring |
| Striatum, ventral tegmental area and substantia nigra [100, 103, 104] | The actor-critic: the critic's value and its prediction error (the dopamine) | | The pistachio (later) and the agreement of views | `ActorCritic`, `arousal=True` |
| Brainstem rhythm for head-bobbing (inventive) | A rhythm added to the neck pitch target | Amplitude, period | Nothing; selection decides whether it exists | New; a gene held at zero |
| Cerebellum, through the medial spiriform nucleus [10] | None in round 1 | | | Later |

The hand-set brain holds about 0.9 million synapses (an estimate; the composing lane counts
them):

- the association reads about 1,200 inputs: two glimpses (278), the two-view units (36), the
  flow units (8), the body's codes (148), and the map's `where` and `what` (728);
- the heard history of one moment adds the same again, and the working trace 256 by 256;
- the gaze field from the association to the two tectal sheets adds about 140,000 at density
  0.25;
- the motor, tectum, fields and categories add the rest.

The recall memory is off in round 1 because the map holds what the first experiment needs. The
owner's multi-head recall joins at the later steps.

### 3.3 Connections

| From | To | Kind | Hand-set | Source |
| --- | --- | --- | --- | --- |
| Left eye | Right tectum; and right eye to left tectum | Fixed, retinotopic | One receptor to one cell's salience | [21] |
| Tectum cells | Tectum cells | Designed: local feedback in a radius, global inhibition | Radius 2 cells; global gain 1 | [14, 15, 20] |
| Tectal winner | Glimpse window | A gate: the fields read only the winner's window | 20 degree window | [15, 20] |
| Glimpse | Association | One-way | Density 1 | [12, 23, 24, 25] |
| Glimpse | Categories | One-way, the categories' own synapses | | [12, 23] |
| Both eyes' overlap | Wulst binocular units | Designed energy units at the winner | 12 disparities | [30, 32, 33] |
| Each eye, a moment and k moments before | Wulst parallax units | Designed | k = 2, 12 shifts | [53, 60, 61] |
| Wulst | Association | One-way; the deep Wulst reaches NCL | Density 1 | [83] |
| Association | Wulst | Reciprocal with the forward projection (NI and the Wulst) | Density 0.25 | [25] |
| Wulst, categories, head direction | Gaze-to-place step (DL) | Inputs of the arithmetic | | [75, 78] |
| Gaze-to-place step | Map | `Map.moment(seen=[(kind, place)], lit=...)` | | [75, 76] |
| Map (`where`, `what`) | Association | One-way | Density 1 | [75] |
| Proprioception, head direction, flow, efference copy | Association | One-way | Density 1 | [39, 79] |
| Working trace | Association | One-way, carried gain | Library founder | [86, 87] |
| Association | Motor slots | Reciprocal (NCL to AI) | Density 1 | [83] |
| Association | Tectum sheet | One-way, applied at the next moment (arcopallial gaze field) | Density 0.25 | [15, 44] |
| Tectal winner | Head yaw and pitch slots | Designed, four directional channels | Gain 1 | [42, 43, 98] |
| Critic | Every plastic synapse of the actor | The prediction error | Library founder | [84, 100, 103] |
| Map | Tectum sheet | Inventive: things held on the map raise salience where they should appear | Gain 0 (gene) | none |

### 3.4 How distance is learned

**The principle: a thing's distance is the number that makes two views of it agree.**

- **Views the parrot has.** The two eyes see a thing in the front overlap from places 3.6 cm
  apart. One eye sees a thing before and after the head moves, from places up to 4.6 cm apart
  (a neck lean) or 2.5 cm apart (a crouch). A head turn moves each eye on a circle of 2.2 cm
  radius about the pivot.
- **The arithmetic.** For a thing at distance d, the image shifts by the baseline over d,
  times the sine of the angle between the baseline and the line of sight. Read with the baseline
  the body knows from its own movement, the shift gives the distance. That is the triangulation
  the brain has to learn. Birds that judge depth this way move their heads [53, 60, 61, 62].

**The two-view units (Wulst).**

- **Disparity.** Binocular units compare the left and right eye's image at the winner, at 12
  disparities from 0 to 22 degrees, as energy-model units [32].
- **Parallax.** Parallax units compare one eye's image at the winner with that eye's image k
  moments before, at 12 shifts.
- **The same rule twice.** Disparity and parallax are one comparison made over two baselines:
  across the eyes, and across time.

**The teachers.** Neither tells a distance, and each acts in the brain's life from the first
moment.

1. **Sight teaches the next view (lessons).**
   - At each moment the next-view slot says how far the thing the parrot looks at will shift
     in the eye after the movement the brain commands.
   - At the next moment, the eye shows where it went. That observed shift is the lesson's
     label.
   - The label comes from the world. The shift of a thing at the side after a lean is the
     lean's length over the thing's distance, so to predict it the association must carry the
     distance. This is the stage the space guide names learned physics on the map (#324), on
     the retina.
2. **Agreement pays (reward).**
   - Each winner has a reach slot, one per eye at the hand-set. The reach slot names a distance
     for the winner's thing, and the gaze-to-place step writes that place on the map. When a
     second view of the same thing writes it again, the map compares the two places.
   - The second view is the other eye, or the same eye after the eyes moved at least 1 cm. The
     reward is `exp(-gap / 2 cm)`.
   - The reward pays the reach act that wrote the place, through the actor-critic's dopamine.
   - For rays from two places, only the true distance makes the places agree. With no
     movement, every distance agrees, so the reward counts only when the baseline is at least
     the declared one.
   - The brain is paid for a map that agrees with itself, which is the Manifesto's local repair
     of disagreement, here on the map.
3. **Later: touch and the pistachio.**
   - A contact sense at the beak tip gives the felt place of a thing within reach.
   - A pistachio taken from the visitor's fingers pays.
   - Both arrive with the body's touch sense (section 3.8).

**The brain's sense of distance.**

- **The reach slot.** Its choice, or the probability-weighted mean of its bins, is the
  distance the brain acts on. It is the page's `readout.distance`, and the fitness and E3
  read it.
- **The distance neurons.** The page draws the neurons whose activity follows distance. The
  chamber's probe finds them over held-out lives. The probe is a measurement and never sits in
  the brain's path.

### 3.5 One moment, in order

1. **Winners.** The eyes render. Each tectum settles its salience under the isthmic competition
   to its winner.
2. **What it is.** The fields read the glimpse at each winner. The categories settle on its form
   and tint, which gives the kind. The motion and looming units read the glimpse against the
   whole field's flow.
3. **Two views.** The Wulst units read the two eyes at the winner (in the overlap) and each eye
   against its view k moments before.
4. **The settle.**
   - The association settles on the glimpses, the Wulst, the body's codes, the efference copy,
     the map's `where` and `what`, and its working trace.
   - The motor settles on the joints, the reach and the next view.
   - The tectal winner adds its orienting drive to the head slots.
5. **The map.** The gaze-to-place step turns each winner's ray and the reach distance into a
   place. `Map.moment(seen=..., lit=...)` holds it, and the map matches it to a thing of its
   kind (within `step`) or enters a new thing.
6. **Learning.**
   - The fields take their step, and the categories take theirs.
   - The next-view slot is taught the shift the eye showed for the previous moment's target.
   - The map's agreement and (later) the pistachio are the reward of the reach and joint acts.
   - The joints move.

The next-view slot learns by lessons, and the joints and reach learn by reward. How one moment
carries both is the composing lane's choice:

- one `step` with a teacher for one slot and an outcome for the others; or
- a second learning head over the next-view slot (`Learner.sharing`, with disjoint masks).

### 3.6 The motor: head, neck, legs, beak, wings

- **Head and gaze.** Head yaw and pitch take 7 bins each, as changes of the target per moment:
  -30, -10, -3, 0, +3, +10 and +30 degrees. Head roll takes 3. The tectum's orienting drive
  turns the head to the winner, and the association's actor learns where to look.
- **Neck pitch** takes 3 bins (back, hold, forward): the lean that peers. The head-bob rhythm
  gene adds to its target.
- **Legs** take 3 bins each (taller, hold, crouch). A crouch moves the eyes down 2.5 cm, a
  vertical baseline.
- **Beak, wings and grips** take 3 bins each. They start biased to rest, closed on the perch for
  the grips. They have no task in the first experiment and are wired so that later tasks
  (taking a pistachio, a foot bringing a thing up to the eye [56]) find them in the brain.
- **Reach** takes 12 log-spaced bins from 0.08 to 1.1 m, one slot per winner (two at the
  hand-set, one for each eye's winner). Each bin is 27% wider than the one before, and the
  probability-weighted mean reads finer than a bin. Birds express distance in the stop of the
  head before a peck [60, 61], so the reach is a motor act.
- **Next view** takes 9 bins of shift, one slot per winner.

**Why the map is told only what the brain looked at.** `lit` names the cells the brain sees, and
a held thing whose place is seen with no report of its kind leaves that place. The brain reports
only its winners, one per eye at the hand-set. With `lit` as the whole field of view, things in
view that were not looked at would be taken off their places. At the hand-set, `lit` is
therefore the cells inside the glimpse windows, out to the distance the brain read there: a
parrot finds a place empty where it looks. The other rule, the whole field of view, is a gene.

### 3.7 What is new to the library

| Part | What it is | Where it lives |
| --- | --- | --- |
| Tectal priority sheet | A designed region: salience in, local feedback and global inhibition, k winners | The parrot's code (`src/cadence/parrot/`) |
| The glimpse | Learned fields read on a window about a winner, so what and where are bound per thing | The parrot's code; it uses `LearnedFields` |
| Foveated pooling in two dimensions | The retina's receptors, denser about the eye's axis | The parrot's code |
| Two-view units | Disparity and parallax units at the winner | The parrot's code |
| The step from gaze to place | Place from the eye's pose, the pixel's ray and the reach distance | The parrot's code |
| The agreement reward | The map's gap between two views of one thing, as a reward | The parrot's code; it reads `Map.things()` |
| The head-bob rhythm | A rhythm on the neck's target | The parrot's code |
| A learned gain-field region (round 2) | Ray and head direction to place, learned from agreement | The parrot's code; a later lane |

Nothing here changes `Map`, `LearnedFields` or a shared core file.

### 3.8 What the brain asks of the body and the chamber

- **A sidestep along the perch (proposal).** A twelfth joint, the body's place along the bar
  (about ±0.10 m at 0.2 m/s), gives baselines of up to 20 cm. That sharpens the parallax tenfold
  at the far edge of the table. Parrots move along branches with beak and feet [99]. This is a
  gene of the body: off in round 1.
- **A touch sense at the beak tip (proposal).** One value: contact. It is the later teacher of
  distance within reach, and the sense the pistachio needs.
- **Eyes fixed in the head.** Real parrots move their eyes up to 24 degrees [56]. The chamber's
  fixed eyes make gaze a matter of the head, as head movement is a proxy for gaze in pigeons
  [45].
- **Panels that remove the cues that tell distance by other means.** Random stands and scales
  remove height in the field and familiar size. A panel of things on the table, read apart,
  shows what height in the field adds.

### 3.9 What the first experiment predicts

| Reading | Prediction from the sources and the geometry |
| --- | --- |
| One eye covered | Distance in the front overlap worse; at the sides about the same, carried by the lean [22, 63] |
| Head held fixed (neck, legs and yaw frozen) | Distance at the sides at chance for a newborn's life; in front, unchanged |
| Wulst silenced | As one eye covered in front, plus the parallax lost |
| Tectum's winner drawn at random | Things found and held, with fewer things looked at twice, and so less agreement and slower distance learning |
| Map off (last seen only) | Things out of view held at their last seen place: no change at rest, worse for a thing moved unseen, as the space guide's readings show |
| Selection raises the head-bob amplitude | Parallax pays, as in birds that bob [58, 59, 61] |
| Selection moves disparity to the rotundus | The pigeon's arrangement serves a lateral-eyed bird [22] |

## 4. The gene space for the genetic algorithm

### 4.1 The genome

The genome has two parts.

- **A dict of named genes for `cadence.genes(space)`.** Each is declared as `("log", step, low,
  high)`, `("linear", step, low, high)`, `("int", low, high)` or `("choice", ...)`.
- **A `Genome` of the learned regions and their projections.** Its region sizes, densities,
  signs and scales change by `cadence.genome.mutate`. The eyes' inputs and the motor's slots are
  fixed, and the working trace is tied to the association it holds.

Every gene starts at its hand-set value, from the sources or the rig's own measure. That
hand-set genome is the lineage's first member and the control. Genes marked F form the first
lineage.

**A. Eye and retina**

| Gene | Declaration | Hand-set | What it sets | Why the hand-set value |
| --- | --- | --- | --- | --- |
| `fovea_azimuth` F | linear, 5, 30, 110 | 75 | Where the sharp zone points, degrees from the beak | The rig's eye axis; sharp regions look sideways [50, 57] |
| `fovea_width` F | log, 0.2, 10, 80 | 30 | The width of the sharp zone, degrees | Designer's choice |
| `fovea_falloff` | linear, 0.1, 0.1, 1.0 | 0.4 | How fast the receptors coarsen | The retina lane's `Eye` |
| `frontal_zone` | choice, 0, 1 | 0 | A second sharp zone toward the beak | Budgerigars have an area centralis without a fovea [57] |
| `change_channel` | choice, 0, 1 | 1 | Change since the moment before, as a channel | Tectal motion cells [26, 40]; the Eyes demo |
| `eyes_tied` | choice, 0, 1 | 1 | One fields dictionary for both eyes | The two retinas develop alike |

**B. Tectum and isthmi**

| Gene | Declaration | Hand-set | What it sets | Why |
| --- | --- | --- | --- | --- |
| `tectum_spacing` F | log, 0.2, 2, 10 | 4 | Degrees a tectal cell | Designer's choice for a 175 by 100 degree eye |
| `isthmi_global` F | log, 0.3, 0.05, 5 | 1.0 | Imc's global inhibition | [14, 15, 20] |
| `isthmi_local` F | log, 0.3, 0.05, 5 | 0.5 | Ipc and SLu local feedback gain | [14, 15, 20] |
| `isthmi_radius` | int, 1, 6 | 2 | Local feedback radius in cells | Ipc fields of 15 to 25 degrees [20] |
| `isthmi_asymmetry` | linear, 0.1, 0, 1 | 0.5 | How much the upper and central field suppress the lower | [20] |
| `winners` F | int, 1, 6 | 1 | Winners per tectum per moment | Winner-take-all [14, 15] |
| `orient_gain` F | log, 0.3, 0.01, 10 | 1.0 | The tectum's drive to the head slots | [42, 43, 98] |
| `gaze_field` | linear, 0.1, 0, 1 | 0.25 | Density of the association to tectum projection | [15, 44] |
| `map_expectation` F | linear, 0.1, 0, 1 | 0 | The map's held things raise salience where they should appear | Inventive; no source |
| `novelty` | linear, 0.1, 0, 1 | 0 | Salience rises with the map's clock at a place | Inventive; tectal novelty coding [14] |

**C. What: rotundus and entopallium**

| Gene | Declaration | Hand-set | What it sets | Why |
| --- | --- | --- | --- | --- |
| `fields` | log, 0.3, 8, 128 (rounded) | 32 | Learned receptive fields | The pixel chamber's range |
| `glimpse_width` F | log, 0.2, 8, 40 | 20 | The window the winner gates, degrees | Large rotundus fields [12, 20] |
| `looming` | choice, 0, 1 | 1 | Tau, rho and eta units | [27, 28] |
| `motion_opponent` | choice, 0, 1 | 1 | Motion against the whole field | [26, 40] |
| `kinds_capacity` | int, 8, 48 | 24 | Categories of kind | The retina lane's founder |
| `kinds_vigilance` | linear, 0.05, 0.2, 0.7 | 0.4 | How unlike a thing must be to take a new category | The retina lane's founder |

**D. How far: the Wulst**

| Gene | Declaration | Hand-set | What it sets | Why |
| --- | --- | --- | --- | --- |
| `disparity_site` F | choice, wulst, rotundus | wulst | Which pathway compares the two eyes | Owls [30, 32]; the pigeon's left rotundus [22] |
| `disparity_form` F | choice, energy, learned | energy | Designed energy units or binocular fields learned by the fields' rule | [32, 33] |
| `disparity_channels` F | int, 4, 24 | 12 | Disparities tuned | Designer's choice |
| `disparity_max` | log, 0.2, 6, 40 | 22 | The largest disparity, degrees | The rig: 2 atan(1.8 cm / 9 cm) |
| `parallax_delay` F | int, 1, 6 | 2 | k: moments between the two views of one eye | Bobbing holds and thrusts [58, 59]; no timing found |
| `parallax_channels` | int, 4, 24 | 12 | Shifts tuned | Designer's choice |
| `ipsilateral_sign` | linear, 0.2, -1, 1 | -0.5 | The sign of the same-side eye's input onto the Wulst's monocular units | Inhibitory in the zebra finch [34] |

**E. Self-motion and the head**

| Gene | Declaration | Hand-set | What it sets | Why |
| --- | --- | --- | --- | --- |
| `hd_units` | int, 8, 36 | 16 | Head-direction units | Broad avian tuning [79] |
| `efference_amplitude` | linear, 0.2, 0, 2 | 1.0 | The efference copy | The library's |
| `flow_units` | choice, 0, 1 | 1 | The optic-flow units | [39, 40] |
| `peer_amplitude` F | linear, 0.1, 0, 1 | 0 | The head-bob rhythm's share of the neck's range | Absent in the control; selection decides [58, 61] |
| `peer_period` F | int, 2, 12 | 6 | Its period, in moments | Designer's choice |
| `perch_walk` | choice, 0, 1 | 0 | The sidestep joint, where the body offers it | [99] |

**F. The map and the step to it**

| Gene | Declaration | Hand-set | What it sets | Why |
| --- | --- | --- | --- | --- |
| `place_step` | choice, geometry, learned | geometry | Designed gaze-to-place step, or the learned gain field (round 2) | [78] |
| `map_cells` | log, 0.2, 16, 64 (rounded) | 32 | Cells along the table's length | 5 cm cells |
| `map_reach` | log, 0.3, 0.05, 0.6 | 0.15 | `Map(reach=)` | The map's founder |
| `match_step` F | log, 0.3, 0.05, 0.4 | 0.15 | `Map(step=)`: a report this near a thing of its kind is that thing | The map's founder |
| `match_stride` | log, 0.3, 0.2, 1.0 | 0.5 | `Map(stride=)` | The map's founder |
| `empty_rule` | choice, near, clear, keep | near | `Map(empty=)` | The map's founder |
| `report_views` F | int, 1, 4 | 1 | Views that agree before a found thing is written | Inventive: a write gate |
| `lit_rule` | choice, glimpse, field | glimpse | Cells the brain counts as seen: inside the glimpse windows out to the distance read, or the whole fields of view out to 1.2 m | Section 3.6 |

**G. NCL and the association**

| Gene | Declaration | Hand-set | What it sets | Why |
| --- | --- | --- | --- | --- |
| `association` F (a `Genome` size) | `mutate`, size_step 0.25 | 256 | The association's neurons | Parrots' many pallial neurons [8] |
| `working_memory_amplitude` | linear, 0.1, 0, 2 | founder | The working trace | [86, 87] |
| `carried_gain` | log, 0.5, 0.001, 0.1 | 0.01 | The carried routes' learned gain | The library's founder |
| `history_decays` | choice, (0.0,), (0.0, 0.5), (0.0, 0.8), () | (0.0,) | Heard-history blocks | The library's founder |
| `recall` | choice, 0, 1 | 0 | The recall memory | Off in round 1 |
| `recall_heads` | int, 1, 4 | 1 | Read heads, where recall is on | Capacity limits [90] |

**H. Arcopallium and the motor**

| Gene | Declaration | Hand-set | What it sets | Why |
| --- | --- | --- | --- | --- |
| `gaze_bins` | choice, 5, 7, 9, 13 | 7 | Bins of head yaw and pitch | The Eyes demo uses 13 |
| `reach_bins` F | int, 6, 24 | 12 | Bins of the reach | 27% a bin from 0.08 to 1.1 m |
| `next_view_bins` | choice, 5, 9, 13 | 9 | Bins of the shift predicted | Designer's choice |
| `motor_lateral` | linear, 0.1, -1, 0 | founder | Competition inside a slot | The library's |

**I. Reward and dopamine**

| Gene | Declaration | Hand-set | What it sets | Why |
| --- | --- | --- | --- | --- |
| `agreement_weight` F | log, 0.5, 0.01, 10 | 1.0 | The agreement reward's scale | Inventive |
| `agreement_scale` | log, 0.3, 0.005, 0.1 | 0.02 | Metres at which the reward falls to 1/e | Two pixels' error at 0.5 m |
| `agreement_views` F | choice, binocular, before-after, both | both | Which second views pay | Inventive |
| `agreement_baseline` | log, 0.3, 0.003, 0.05 | 0.01 | The least eye movement for a before-after view to pay | Inventive |
| `predict_weight` F | log, 0.5, 0.01, 10 | 1.0 | The next-view lessons' weight | The space guide's learned physics |
| `actor_eta` | log, 0.3, 0.01, 1 | founder | The actor's rate | [101, 103] |
| `actor_lam` | linear, 0.1, 0, 1 | founder | Eligibility | [101] |
| `arousal` | choice, 0, 1 | 1 | Learn when surprised or young | The library's creature |

**J. Which projections exist**

These are choices of 0 or 1. Each projection's density, sign and scale also move under `mutate`.

| Gene | Hand-set | Projection | Source |
| --- | --- | --- | --- |
| `p_glimpse_dl` | 1 | Entopallium (NFL, NIL) to DL | [78] |
| `p_wulst_dl` | 1 | Hyperpallium densocellulare to DL | [75, 78] |
| `p_ni_wulst` | 1 | Association to Wulst, reciprocal | [25] |
| `p_map_motor` | 1 | Map to motor (DM to arcopallium) | [75] |
| `p_glimpse_motor` | 1 | Glimpse to motor at low density (entopallium to AI) | [24] |
| `p_wulst_tectum` | 0 | Wulst to tectum | Not found in what was read |

That makes 67 named genes, 22 of them F.

### 4.2 Fitness for the first experiment

The chamber computes every term from its per-pixel truth, which never enters the brain. Terms are
read over the last third of each life.

| Term | Measure | Weight |
| --- | --- | --- |
| m1, seen | Median distance on the table between each thing in view and the map's place for it, cm | 1 |
| m2, held | The same for things out of view: behind, in the blind sector, or after the head turned away | 1 |
| m3, one thing | Share of things not held as exactly one thing: duplicates, merges, lost | 0.5 |
| m4, distance | Median of the absolute log ratio of the reach distance to the true distance, over things looked at | 1 |
| m5, surprise | Mean next-view error, degrees | 0.25 |
| cost | Mean multiply-accumulates a moment, with every settle and sweep counted | 0.2 |

```text
fitness = - sum_i w_i * m_i / m_i(hand-set)  -  0.2 * cost / cost(hand-set)
```

- **Scale.** Each term is divided by the hand-set genome's value on the same worlds, so the
  hand-set genome scores -3.95 and every unit of gain is comparable.
- **Surprise is priced beside compute.** A price on compute alone favours genomes that switch
  learning off, as the anatomy guide's selection protocol warns.
- **No gene can reweight its own fitness.** The weights and the scales are not genes.
- **Covered-eye deficit.** The E3 deficit with one eye covered is read on the selected genome
  and never enters the fitness, so selection cannot buy it.

### 4.3 Operators and budget

- **Selection.** `cd.evolve(fitness, hand_set, mutate=mutate, generations=24, population=16,
  keep=4, seed=..., mapper=pool.map)`. Each generation scores 16 offspring of parents drawn from
  the 4 best of the generation before. Mutation and truncation are the library's operators.
- **Mutation.** One custom `mutate(genome, rng)`:
  - `cadence.genes(space, rate=0.15)` on the named genes (about three of the 22 change per
    offspring in the first lineage);
  - `cadence.genome.mutate(..., size_step=0.25, fixed=(eyes, motor), tied=(("association",
    "prefrontal"),))` on the regions and projections.
- **No crossover in round 1.** A block crossover that swaps whole groups (A to J) between two
  parents is new to the library and a round 2 option.
- **Common worlds.** Every genome of a generation lives the same three worlds. A world's seed
  is the life's seed divided by 1,000, which `evolve` makes the generation's: life `i` of
  generation `g` gets `seed + 1000 g + i`. Differences within a generation are then the
  genome's.
- **Worlds.**
  - A world is a life of 2,400 moments (four minutes of the parrot's time).
  - 4 to 8 things are placed, moved and removed by a declared script.
  - Things stand at random heights and scales. Some are placed behind the parrot.
- **Budget.**
  - 24 generations of 16 genomes, each living 3 worlds, make 1,152 lives.
  - At an estimated 30 to 60 ms a moment on one core, that is 23 to 46 core-hours: about 1.5 to
    3 hours on a 16-core box.
  - Random search at equal evaluations costs the same again.
  - The composing lane measures the cost of a moment first and scales the budget to it.
- **Development before the gated run.** A smoke lineage of 3 generations of 4 on one world,
  marked as development.

### 4.4 Controls, and the selection protocol

| Arm | What it is |
| --- | --- |
| Hand-set wiring | Every gene at its hand-set value. The lineage's first member, and the control the selected genome must beat |
| Random wiring of the same size | The same regions at the same sizes, the same number of projections and synapses. Each projection's two ends are redrawn among the region pairs the fixed ends allow, and densities are rescaled to hold the synapse count. Plus the shuffled form: every synapse's receiving neuron permuted, as the protocol instrument's shuffled control |
| Random search | Uniform draws over the same gene space, as many as the lineage scored; the best on the selection worlds |
| Distance given (ceiling, a control only) | The place from the chamber's true distance, to show what the map reaches when distance is perfect |
| Lesions of the selected genome | One eye covered; head held fixed; Wulst silenced; winner drawn at random; map off |
| The image networks of E5 | A convolutional network and a vision transformer on the same images and lives, at matched parameters (the composing lane's) |

The protocol, from the anatomy guide:

1. The selected genome is read on held-out worlds and twelve fresh founders, never on the
   worlds it was selected on.
2. The claim is only what it gains over the hand-set wiring and over random search.
3. Every table carries parameters and multiply-accumulates.
4. Every outcome is kept, including genomes that prune what the sources put in.

### 4.5 Reading the lineage

The lineage says something about birds as well as about the parrot. Each record keeps the best
genome per generation (`Lineage.generations`).

- **The head bobs.** If `peer_amplitude` rises from zero, a rhythm of the neck paid for itself
  in distance.
- **Disparity moves.** If `disparity_site` moves to the rotundus, the pigeon's arrangement suits
  a lateral-eyed bird.
- **The isthmi let more through.** If `winners` rises above one, a single winner loses things
  that the map needs.
- **Memory guides attention.** If `map_expectation` rises, things held on the map guide the
  eyes.

## 5. Later steps, wired from the same sources

| Bird region | Cadence part | Source |
| --- | --- | --- |
| Cochlear nuclei, NL, MLd, nucleus ovoidalis | The speech front end, `cadence.speech.frontend.Hearing` (log-mel every 10 ms), fixed. The time difference between two ears gives a sound's direction, reported to the map as `heard=` | [114, 116, 117] |
| Field L2, then L1 and L3, then the caudomedial nidopallium and caudal mesopallium | Learned fields on the spectrogram, then an auditory association region that holds heard words (the hearing lane, #329) | [112, 115, 118, 120] |
| The tectum teaching auditory space | Sight calibrates where the map places a heard thing | [122, 123] |
| NIDL, where hearing and sight meet | A region where a heard word and a seen thing settle together: the concept both forms reach (#329 H3) | [119] |
| Vocal core: NLC core to AAC core to the motor neurons | The voice's motor: letters said in time (#302). Only the core reaches the motor | [127, 128, 131, 133] |
| Vocal shell: NLC, AAC, MO and NAO shells | A region that holds the word to be said, bound to the concept, inside the pallium with no motor output of its own. A shell lesion shrinks the repertoire, so the shell holds which words the parrot has | [126, 127] |
| Anterior loop: MO and NAO to MMSt to DMm, back to NAO | The reward-taught loop of the voice. The dopamine of a right answer (the pistachio) reaches the striatum. A second critic carries the error of the said word against the plan, heard back | [103, 104, 107, 108, 129] |
| Hippocampal barcodes, episodic-like memory | Long-term records written per event, for "where was the ball?" (#326 W8 to W10) | [74, 77] |
| The map's clock | Moments since a thing was seen: the present against the past | [76, 80] |
| Medial spiriform nucleus and cerebellum | Fine timing of beak, tongue and foot, for taking a pistachio | [10, 95, 99] |

## Sources

Reviews come first in each group.

**The pallium**

1. Jarvis ED, Güntürkün O, Bruce L, et al. (2005). Avian brains and a new understanding of vertebrate brain evolution. *Nature Reviews Neuroscience* 6(2):151-159. https://doi.org/10.1038/nrn1606
2. Reiner A, Perkel DJ, Bruce LL, et al. (2004). Revised nomenclature for avian telencephalon and some related brainstem nuclei. *Journal of Comparative Neurology* 473(3):377-414. https://doi.org/10.1002/cne.20118
3. Güntürkün O, Bugnyar T (2016). Cognition without cortex. *Trends in Cognitive Sciences* 20(4):291-303. https://doi.org/10.1016/j.tics.2016.02.001
4. Güntürkün O, von Eugen K, Packheiser J, Pusch R (2021). Avian pallial circuits and cognition: a comparison to mammals. *Current Opinion in Neurobiology* 71:29-36. https://doi.org/10.1016/j.conb.2021.08.007
5. Jarvis ED, Yu J, Rivas MV, et al. (2013). Global view of the functional molecular organization of the avian cerebrum: mirror images and functional columns. *Journal of Comparative Neurology* 521(16):3614-3665. https://doi.org/10.1002/cne.23404
6. Stacho M, Herold C, Rook N, Wagner H, Axer M, Amunts K, Güntürkün O (2020). A cortex-like canonical circuit in the avian forebrain. *Science* 369(6511):eabc5534. https://doi.org/10.1126/science.abc5534
7. Shanahan M, Bingman VP, Shimizu T, Wild M, Güntürkün O (2013). Large-scale network organization in the avian forebrain: a connectivity matrix and theoretical analysis. *Frontiers in Computational Neuroscience* 7:89. https://doi.org/10.3389/fncom.2013.00089
8. Olkowicz S, Kocourek M, Lučan RK, Porteš M, Fitch WT, Herculano-Houzel S, Němec P (2016). Birds have primate-like numbers of neurons in the forebrain. *PNAS* 113(26):7255-7260. https://doi.org/10.1073/pnas.1517131113
9. Iwaniuk AN, Dean KM, Nelson JE (2005). Interspecific allometry of the brain and brain regions in parrots (Psittaciformes). *Brain, Behavior and Evolution* 65(1):40-59. https://doi.org/10.1159/000081110
10. Gutiérrez-Ibáñez C, Iwaniuk AN, Wylie DR (2018). Parrots have evolved a primate-like telencephalic-midbrain-cerebellar circuit. *Scientific Reports* 8:9960. https://doi.org/10.1038/s41598-018-28301-4

**Vision: pathways**

11. Wylie DR, Gutiérrez-Ibáñez C, Iwaniuk AN (2015). Integrating brain, behavior, and phylogeny to understand the evolution of sensory systems in birds. *Frontiers in Neuroscience* 9:281. https://doi.org/10.3389/fnins.2015.00281
12. Shimizu T, Patton TB, Husband SA (2010). Avian visual behavior and the organization of the telencephalon. *Brain, Behavior and Evolution* 75(3):204-217. https://doi.org/10.1159/000314283
13. Wylie DR, Gutierrez-Ibanez C, Pakan JM, Iwaniuk AN (2009). The optic tectum of birds: mapping our way to understanding visual processing. *Canadian Journal of Experimental Psychology* 63(4):328-338. https://doi.org/10.1037/a0016826
14. Mysore SP, Knudsen EI (2011). The role of a midbrain network in competitive stimulus selection. *Current Opinion in Neurobiology* 21(4):653-660. https://doi.org/10.1016/j.conb.2011.05.024
15. Knudsen EI (2018). Neural circuits that mediate selective attention: a comparative perspective. *Trends in Neurosciences* 41(11):789-805. https://doi.org/10.1016/j.tins.2018.06.006
16. Luksch H (2003). Cytoarchitecture of the avian optic tectum: neuronal substrate for cellular computation. *Reviews in the Neurosciences* 14(1-2):85-106. https://doi.org/10.1515/revneuro.2003.14.1-2.85
17. Medina L, Reiner A (2000). Do birds possess homologues of mammalian primary visual, somatosensory and motor cortices? *Trends in Neurosciences* 23(1):1-12. https://doi.org/10.1016/S0166-2236(99)01486-1
18. Shimizu T, Bowers AN (1999). Visual circuits of the avian telencephalon: evolutionary implications. *Behavioural Brain Research* 98(2):183-191. https://doi.org/10.1016/S0166-4328(98)00083-7
19. Bischof HJ, Watanabe S (1997). On the structure and function of the tectofugal visual pathway in laterally eyed birds. *European Journal of Morphology* 35(4):246-254. https://doi.org/10.1076/ejom.35.4.246.13080
20. Marín G, Salas C, Sentis E, Rojas X, Letelier JC, Mpodozis J (2007). A cholinergic gating mechanism controlled by competitive interactions in the optic tectum of the pigeon. *Journal of Neuroscience* 27(30):8112-8121. https://doi.org/10.1523/JNEUROSCI.1420-07.2007
21. Remy M, Güntürkün O (1991). Retinal afferents to the tectum opticum and the nucleus opticus principalis thalami in the pigeon. *Journal of Comparative Neurology* 305(1):57-70. https://doi.org/10.1002/cne.903050107
22. Güntürkün O, Hahmann U (1999). Functional subdivisions of the ascending visual pathways in the pigeon. *Behavioural Brain Research* 98(2):193-201. https://doi.org/10.1016/S0166-4328(98)00084-9
23. Krützfeldt NO, Wild JM (2005). Definition and novel connections of the entopallium in the pigeon (*Columba livia*). *Journal of Comparative Neurology* 490(1):40-56. https://doi.org/10.1002/cne.20627
24. Husband SA, Shimizu T (1999). Efferent projections of the ectostriatum in the pigeon (*Columba livia*). *Journal of Comparative Neurology* 406(3):329-345. PubMed 10102499
25. Fernández M, Ahumada-Galleguillos P, Sentis E, Marín G, Mpodozis J (2020). Intratelencephalic projections of the avian visual dorsal ventricular ridge: laminarly segregated, reciprocally and topographically organized. *Journal of Comparative Neurology* 528(2):321-359. https://doi.org/10.1002/cne.24757
26. Frost BJ, Nakayama K (1983). Single visual neurons code opposing motion independent of direction. *Science* 220:744-745. https://doi.org/10.1126/science.6836313
27. Wang Y, Frost BJ (1992). Time to collision is signalled by neurons in the nucleus rotundus of pigeons. *Nature* 356:236-238. https://doi.org/10.1038/356236a0
28. Sun H, Frost BJ (1998). Computation of different optical variables of looming objects in pigeon nucleus rotundus neurons. *Nature Neuroscience* 1:296-303. https://doi.org/10.1038/1110
29. Iwaniuk AN, Gutierrez-Ibanez C, Pakan JM, Wylie DR (2010). Allometric scaling of the tectofugal pathway in birds. *Brain, Behavior and Evolution* 75(2):122-137. https://doi.org/10.1159/000311729

**Vision: the Wulst and disparity**

30. Pettigrew JD, Konishi M (1976). Neurons selective for orientation and binocular disparity in the visual Wulst of the barn owl (*Tyto alba*). *Science* 193:675-678. https://doi.org/10.1126/science.948741
31. Pettigrew JD (1979). Binocular visual processing in the owl's telencephalon. *Proceedings of the Royal Society B* 204:435-454. https://doi.org/10.1098/rspb.1979.0038
32. Nieder A, Wagner H (2000). Horizontal-disparity tuning of neurons in the visual forebrain of the behaving barn owl. *Journal of Neurophysiology* 83(5):2967-2979. https://doi.org/10.1152/jn.2000.83.5.2967
33. Nieder A, Wagner H (2001). Hierarchical processing of horizontal disparity information in the visual forebrain of behaving owls. *Journal of Neuroscience* 21(12):4514-4522. https://doi.org/10.1523/JNEUROSCI.21-12-04514.2001
34. Michael N, Löwel S, Bischof HJ (2015). Features of the retinotopic representation in the visual wulst of a laterally eyed bird, the zebra finch (*Taeniopygia guttata*). *PLoS ONE* 10(4):e0124917. https://doi.org/10.1371/journal.pone.0124917
35. Bischof HJ, Eckmeier D, Keary N, Löwel S, Mayer U, Michael N (2016). Multiple visual field representations in the visual wulst of a laterally eyed bird, the zebra finch (*Taeniopygia guttata*). *PLoS ONE* 11(5):e0154927. https://doi.org/10.1371/journal.pone.0154927
36. Iwaniuk AN, Heesy CP, Hall MI, Wylie DR (2008). Relative Wulst volume is correlated with orbit orientation and binocular visual field in birds. *Journal of Comparative Physiology A* 194(3):267-282. https://doi.org/10.1007/s00359-007-0304-0
37. Reiner A, Karten HJ (1983). The laminar source of efferent projections from the avian Wulst. *Brain Research* 275(2):349-354. https://doi.org/10.1016/0006-8993(83)90996-4
38. Wild JM, Williams MN (2000). Rostral Wulst in passerine birds. I. Origin, course, and terminations of an avian pyramidal tract. *Journal of Comparative Neurology* 416(4):429-450. https://doi.org/10.1002/(SICI)1096-9861(20000124)416:4%3C429::AID-CNE2%3E3.0.CO;2-X

**Self-motion, gaze and the head**

39. Wylie DR (2013). Processing of visual signals related to self-motion in the cerebellum of pigeons. *Frontiers in Behavioral Neuroscience* 7:4. https://doi.org/10.3389/fnbeh.2013.00004
40. Frost BJ, Wylie DR, Wang YC (1990). The processing of object and self-motion in the tectofugal and accessory optic pathways of birds. *Vision Research* 30(11):1677-1688. https://doi.org/10.1016/0042-6989(90)90152-B
41. Wylie DR, Ogilvie CJ, Crowder NA, Barkley RR, Winship IR (2005). Telencephalic projections to the nucleus of the basal optic root and pretectal nucleus lentiformis mesencephali in pigeons. *Visual Neuroscience* 22(2):237-247. https://doi.org/10.1017/S0952523805221090
42. Masino T, Knudsen EI (1990). Horizontal and vertical components of head movement are controlled by distinct neural circuits in the barn owl. *Nature* 345:434-437. https://doi.org/10.1038/345434a0
43. Masino T, Knudsen EI (1993). Orienting head movements resulting from electrical microstimulation of the brainstem tegmentum in the barn owl. *Journal of Neuroscience* 13(1):351-370. https://doi.org/10.1523/JNEUROSCI.13-01-00351.1993
44. Knudsen EI, Cohen YE, Masino T (1995). Characterization of a forebrain gaze field in the archistriatum of the barn owl. *Journal of Neuroscience* 15(7):5139-5151. https://doi.org/10.1523/JNEUROSCI.15-07-05139.1995
45. Kano F, Walker J, Sasaki T, Biro D (2018). Head-mounted sensors reveal visual attention of free-flying homing pigeons. *Journal of Experimental Biology* 221:jeb183475. https://doi.org/10.1242/jeb.183475
46. Theunissen LM, Troje NF (2017). Head stabilization in the pigeon: role of vision to correct for translational and rotational disturbances. *Frontiers in Neuroscience* 11:551. https://doi.org/10.3389/fnins.2017.00551
47. Gioanni H (1988). Stabilizing gaze reflexes in the pigeon (*Columba livia*). II. Vestibulo-ocular (VOR) and vestibulo-collic (closed-loop VCR) reflexes. *Experimental Brain Research* 69(3):583-593. https://doi.org/10.1007/BF00247311
48. Haque A, Dickman JD (2005). Vestibular gaze stabilization: different behavioral strategies for arboreal and terrestrial avians. *Journal of Neurophysiology* 93(3):1165-1173. https://doi.org/10.1152/jn.00966.2004

**Visual fields and distance**

49. Martin GR (2014). The subtlety of simple eyes: the tuning of visual fields to perceptual challenges in birds. *Philosophical Transactions of the Royal Society B* 369:20130040. https://doi.org/10.1098/rstb.2013.0040
50. Martin GR (2017). What drives bird vision? Bill control and predator detection overshadow flight. *Frontiers in Neuroscience* 11:619. https://doi.org/10.3389/fnins.2017.00619
51. Martin GR (2009). What is binocular vision for? A birds' eye view. *Journal of Vision* 9(11):14. https://doi.org/10.1167/9.11.14
52. Martin GR (2007). Visual fields and their functions in birds. *Journal of Ornithology* 148(Suppl 2):S547-S562. https://doi.org/10.1007/s10336-007-0213-6
53. Kral K (2003). Behavioural-analytical studies of the role of head movements in depth perception in insects, birds and mammals. *Behavioural Processes* 64:1-12. https://doi.org/10.1016/S0376-6357(03)00054-8
54. Necker R (2007). Head-bobbing of walking birds. *Journal of Comparative Physiology A* 193:1177-1183. https://doi.org/10.1007/s00359-007-0281-3
55. Glasser A, Howland HC (1996). A history of studies of visual accommodation in birds. *Quarterly Review of Biology* 71(4):475-509. https://doi.org/10.1086/419554
56. Demery ZP, Chappell J, Martin GR (2011). Vision, touch and object manipulation in Senegal parrots *Poicephalus senegalus*. *Proceedings of the Royal Society B* 278(1725):3687-3693. https://doi.org/10.1098/rspb.2011.0374
57. Mitkus M, Chaib S, Lind O, Kelber A (2014). Retinal ganglion cell topography and spatial resolution of two parrot species: budgerigar (*Melopsittacus undulatus*) and Bourke's parrot (*Neopsephotus bourkii*). *Journal of Comparative Physiology A* 200(5):371-384. https://doi.org/10.1007/s00359-014-0894-2
58. Frost BJ (1978). The optokinetic basis of head-bobbing in the pigeon. *Journal of Experimental Biology* 74:187-195. https://doi.org/10.1242/jeb.74.1.187
59. Troje NF, Frost BJ (2000). Head-bobbing in pigeons: how stable is the hold phase? *Journal of Experimental Biology* 203:935-940. https://doi.org/10.1242/jeb.203.5.935
60. Theunissen LM, Reid T, Troje NF (2017). Pigeons use distinct stop phases to control pecking. *Journal of Experimental Biology* 220:437-444. https://doi.org/10.1242/jeb.147850
61. Hataji Y, Kuroshima H, Fujita K (2021). Motion parallax via head movements modulates visuo-motor control in pigeons. *Journal of Experimental Biology* 224(3):jeb236547. https://doi.org/10.1242/jeb.236547
62. Ohayon S, van der Willigen RF, Wagner H, Katsman I, Rivlin E (2006). On the barn owl's visual pre-attack behavior: I. Structure of head movements and motion patterns. *Journal of Comparative Physiology A* 192(9):927-940. https://doi.org/10.1007/s00359-006-0130-9
63. McFadden SA, Wild JM (1986). Binocular depth perception in the pigeon. *Journal of the Experimental Analysis of Behavior* 45:149-160. https://doi.org/10.1901/jeab.1986.45-149
64. McFadden SA (1987). The binocular depth stereoacuity of the pigeon and its relation to the anatomical resolving power of the eye. *Vision Research* 27(11):1967-1980. https://doi.org/10.1016/0042-6989(87)90061-7
65. van der Willigen RF (2011). Owls see in stereo much like humans do. *Journal of Vision* 11(7):10. https://doi.org/10.1167/11.7.10
66. Fox R, Lehmkuhle SW, Bush RC (1977). Stereopsis in the falcon. *Science* 197:79-81. https://doi.org/10.1126/science.867054
67. Schaeffel F, Howland HC (1987). Corneal accommodation in chick and pigeon. *Journal of Comparative Physiology A* 160:375-384. https://doi.org/10.1007/BF00613027
68. Ortega LJ, Stoppa K, Güntürkün O, Troje NF (2009). Vision during head bobbing: are pigeons capable of shape discrimination during the thrust phase? *Experimental Brain Research* 199:313-321. https://doi.org/10.1007/s00221-009-1891-5

**The hippocampal formation**

69. Sherry DF, Jacobs LF, Gaulin SJ (1992). Spatial memory and adaptive specialization of the hippocampus. *Trends in Neurosciences* 15(8):298-303. https://doi.org/10.1016/0166-2236(92)90080-R
70. Atoji Y, Wild JM (2006). Anatomy of the avian hippocampal formation. *Reviews in the Neurosciences* 17(1-2):3-15. https://doi.org/10.1515/revneuro.2006.17.1-2.3
71. Bingman VP, Gagliardo A, Hough GE, Ioalé P, Kahn MC, Siegel JJ (2005). The avian hippocampus, homing in pigeons and the memory representation of large-scale space. *Integrative and Comparative Biology* 45(3):555-564. https://doi.org/10.1093/icb/45.3.555
72. Mayer U, Watanabe S, Bischof HJ (2013). Spatial memory and the avian hippocampus: research in zebra finches. *Journal of Physiology Paris* 107(1-2):2-12. https://doi.org/10.1016/j.jphysparis.2012.05.002
73. Krebs JR, Sherry DF, Healy SD, Perry VH, Vaccarino AL (1989). Hippocampal specialization of food-storing birds. *PNAS* 86(4):1388-1392. https://doi.org/10.1073/pnas.86.4.1388
74. Clayton NS, Dickinson A (1998). Episodic-like memory during cache recovery by scrub jays. *Nature* 395(6699):272-274. https://doi.org/10.1038/26216
75. Atoji Y, Wild JM (2004). Fiber connections of the hippocampal formation and septum and subdivisions of the hippocampal formation in the pigeon as revealed by tract tracing and kainic acid lesions. *Journal of Comparative Neurology* 475(3):426-461. https://doi.org/10.1002/cne.20186
76. Payne HL, Lynch GF, Aronov D (2021). Neural representations of space in the hippocampus of a food-caching bird. *Science* 373(6552):343-348. https://doi.org/10.1126/science.abg2009
77. Chettih SN, Mackevicius EL, Hale S, Aronov D (2024). Barcoding of episodic memories in the hippocampus of a food-caching bird. *Cell* 187(8):1922-1935.e20. https://doi.org/10.1016/j.cell.2024.02.032
78. Applegate MC, Gutnichenko KS, Mackevicius EL, Aronov D (2023). An entorhinal-like region in food-caching birds. *Current Biology* 33(12):2465-2477.e7. https://doi.org/10.1016/j.cub.2023.05.031
79. Ben-Yishay E, Krivoruchko K, Ron S, Ulanovsky N, Derdikman D, Gutfreund Y (2021). Directional tuning in the hippocampal formation of birds. *Current Biology* 31(12):2592-2602.e4. https://doi.org/10.1016/j.cub.2021.04.029
80. Agarwal A, Sarel A, Derdikman D, Ulanovsky N, Gutfreund Y (2023). Spatial coding in the hippocampus and hyperpallium of flying owls. *PNAS* 120(5):e2212418120. https://doi.org/10.1073/pnas.2212418120

**The caudolateral nidopallium**

81. Güntürkün O (2005). The avian 'prefrontal cortex' and cognition. *Current Opinion in Neurobiology* 15(6):686-693. https://doi.org/10.1016/j.conb.2005.10.003
82. Durstewitz D, Kröner S, Güntürkün O (1999). The dopaminergic innervation of the avian telencephalon. *Progress in Neurobiology* 59(2):161-195. https://doi.org/10.1016/S0301-0082(98)00100-2
83. Kröner S, Güntürkün O (1999). Afferent and efferent connections of the caudolateral neostriatum in the pigeon (*Columba livia*): a retro- and anterograde pathway tracing study. *Journal of Comparative Neurology* 407(2):228-260. https://doi.org/10.1002/(SICI)1096-9861(19990503)407:2%3C228::AID-CNE6%3E3.0.CO;2-2
84. Waldmann C, Güntürkün O (1993). The dopaminergic innervation of the pigeon caudolateral forebrain: immunocytochemical evidence for a 'prefrontal cortex' in birds? *Brain Research* 600(2):225-234. https://doi.org/10.1016/0006-8993(93)91377-5
85. von Eugen K, Tabrik S, Güntürkün O, Ströckens F (2020). A comparative analysis of the dopaminergic innervation of the executive caudal nidopallium in pigeon, chicken, zebra finch, and carrion crow. *Journal of Comparative Neurology* 528(17):2929-2955. https://doi.org/10.1002/cne.24878
86. Diekamp B, Kalt T, Güntürkün O (2002). Working memory neurons in pigeons. *Journal of Neuroscience* 22(4):RC210. https://doi.org/10.1523/JNEUROSCI.22-04-j0002.2002
87. Rose J, Colombo M (2005). Neural correlates of executive control in the avian brain. *PLoS Biology* 3(6):e190. https://doi.org/10.1371/journal.pbio.0030190
88. Veit L, Nieder A (2013). Abstract rule neurons in the endbrain support intelligent behaviour in corvid songbirds. *Nature Communications* 4:2878. https://doi.org/10.1038/ncomms3878
89. Veit L, Hartmann K, Nieder A (2014). Neuronal correlates of visual working memory in the corvid endbrain. *Journal of Neuroscience* 34(23):7778-7786. https://doi.org/10.1523/JNEUROSCI.0612-14.2014
90. Hahn LA, Balakhonov D, Fongaro E, Nieder A, Rose J (2021). Working memory capacity of crows and monkeys arises from similar neuronal computations. *eLife* 10:e72783. https://doi.org/10.7554/eLife.72783

**The arcopallium and the motor output**

91. Mello CV, Kaser T, Buckner AA, Wirthlin M, Lovell PV (2019). Molecular architecture of the zebra finch arcopallium. *Journal of Comparative Neurology* 527(15):2512-2556. https://doi.org/10.1002/cne.24688
92. Zeier H, Karten HJ (1971). The archistriatum of the pigeon: organization of afferent and efferent connections. *Brain Research* 31(2):313-326. https://doi.org/10.1016/0006-8993(71)90185-5 (cited as summarised in [91])
93. Davies DC, Csillag A, Székely AD, Kabai P (1997). Efferent connections of the domestic chick archistriatum: a phaseolus lectin anterograde tracing study. *Journal of Comparative Neurology* 389(4):679-693. https://doi.org/10.1002/(SICI)1096-9861(19971229)389:4%3C679::AID-CNE10%3E3.0.CO;2-7
94. Fernández M, Morales C, Durán E, Fernández-Colleman S, Sentis E, Mpodozis J, Karten HJ, Marín GJ (2020). Parallel organization of the avian sensorimotor arcopallium: tectofugal visual pathway in the pigeon (*Columba livia*). *Journal of Comparative Neurology* 528(4):597-623. https://doi.org/10.1002/cne.24775
95. Wild JM, Farabaugh SM (1996). Organization of afferent and efferent projections of the nucleus basalis prosencephali in a passerine, *Taeniopygia guttata*. *Journal of Comparative Neurology* 365(2):306-328. https://doi.org/10.1002/(SICI)1096-9861(19960205)365:2%3C306::AID-CNE8%3E3.0.CO;2-9
96. Dubbeldam JL (1998). The neural substrate for 'learned' and 'nonlearned' activities in birds: a discussion of the organization of bulbar reticular premotor systems with side-lights on the mammalian situation. *Acta Anatomica* 163(3):157-172. https://doi.org/10.1159/000046494
97. Tellegen AJ, Dubbeldam JL (1999). Location of reticular premotor areas of a motor center innervating craniocervical muscles in the mallard (*Anas platyrhynchos* L.). *Journal of Comparative Neurology* 405(3):281-298. https://doi.org/10.1002/(SICI)1096-9861(19990315)405:3%3C281::AID-CNE1%3E3.0.CO;2-Y
98. Tellegen AJ, Karssen AM, Rietveld TM, Dubbeldam JL (1998). A crossed projection from the optic tectum to craniocervical premotor areas in the brainstem reticular formation. An anterograde and retrograde tracing study in the mallard (*Anas platyrhynchos* L.). *European Journal of Morphology* 36(4):227-243. https://doi.org/10.1076/ejom.36.4.227.5823
99. Young MW, Dickinson E, Flaim ND, Granatosky MC (2022). Overcoming a 'forbidden phenotype': the parrot's head supports, propels and powers tripedal locomotion. *Proceedings of the Royal Society B* 289(1975):20220245. https://doi.org/10.1098/rspb.2022.0245

**The basal ganglia and dopamine**

100. Kuenzel WJ, Medina L, Csillag A, Perkel DJ, Reiner A (2011). The avian subpallium: new insights into structural and functional subdivisions occupying the lateral subpallial wall and their embryological origins. *Brain Research* 1424:67-101. https://doi.org/10.1016/j.brainres.2011.09.037
101. Fee MS, Goldberg JH (2011). A hypothesis for basal ganglia-dependent reinforcement learning in the songbird. *Neuroscience* 198:152-170. https://doi.org/10.1016/j.neuroscience.2011.09.069
102. Kubikova L, Košťál L (2010). Dopaminergic system in birdsong learning and maintenance. *Journal of Chemical Neuroanatomy* 39(2):112-123. https://doi.org/10.1016/j.jchemneu.2009.10.004
103. Gadagkar V, Puzerey PA, Chen R, Baird-Daniel E, Farhang AR, Goldberg JH (2016). Dopamine neurons encode performance error in singing birds. *Science* 354(6317):1278-1282. https://doi.org/10.1126/science.aah6837
104. Roeser A, Gadagkar V, Das A, Puzerey PA, Kardon B, Goldberg JH (2023). Dopaminergic error signals retune to social feedback during courtship. *Nature* 623(7986):375-380. https://doi.org/10.1038/s41586-023-06580-w
105. Gale SD, Perkel DJ (2010). A basal ganglia pathway drives selective auditory responses in songbird dopaminergic neurons via disinhibition. *Journal of Neuroscience* 30(3):1027-1037. https://doi.org/10.1523/JNEUROSCI.3585-09.2010
106. Ding L, Perkel DJ (2004). Long-term potentiation in an avian basal ganglia nucleus essential for vocal learning. *Journal of Neuroscience* 24(2):488-494. https://doi.org/10.1523/JNEUROSCI.4358-03.2004
107. Xiao L, Chattree G, Oscos FG, Cao M, Wanat MJ, Roberts TF (2018). A basal ganglia circuit sufficient to guide birdsong learning. *Neuron* 98(1):208-221.e5. https://doi.org/10.1016/j.neuron.2018.02.020
108. Hisey E, Kearney MG, Mooney R (2018). A common neural circuit mechanism for internally guided and externally reinforced forms of motor learning. *Nature Neuroscience* 21(4):589-597. https://doi.org/10.1038/s41593-018-0092-6
109. Yanagihara S, Izawa E, Koga K, Matsushima T (2001). Reward-related neuronal activities in basal ganglia of domestic chicks. *NeuroReport* 12(7):1431-1435. https://doi.org/10.1097/00001756-200105250-00027
110. Izawa E, Zachar G, Yanagihara S, Matsushima T (2003). Localized lesion of caudal part of lobus parolfactorius caused impulsive choice in the domestic chick: evolutionarily conserved function of ventral striatum. *Journal of Neuroscience* 23(5):1894-1902. https://doi.org/10.1523/JNEUROSCI.23-05-01894.2003
111. Izawa E, Aoki N, Matsushima T (2005). Neural correlates of the proximity and quantity of anticipated food rewards in the ventral striatum of domestic chicks. *European Journal of Neuroscience* 22(6):1502-1512. https://doi.org/10.1111/j.1460-9568.2005.04311.x

**Hearing**

112. Theunissen FE, Shaevitz SS (2006). Auditory processing of vocal sounds in birds. *Current Opinion in Neurobiology* 16(4):400-407. https://doi.org/10.1016/j.conb.2006.07.003
113. Konishi M (2003). Coding of auditory space. *Annual Review of Neuroscience* 26:31-55. https://doi.org/10.1146/annurev.neuro.26.041002.131123
114. Ashida G, Carr CE (2011). Sound localization: Jeffress and beyond. *Current Opinion in Neurobiology* 21(5):745-751. https://doi.org/10.1016/j.conb.2011.05.008
115. Bolhuis JJ, Gahr M (2006). Neural mechanisms of birdsong memory. *Nature Reviews Neuroscience* 7(5):347-357. https://doi.org/10.1038/nrn1904
116. Carr CE, Konishi M (1990). A circuit for detection of interaural time differences in the brain stem of the barn owl. *Journal of Neuroscience* 10(10):3227-3246. https://doi.org/10.1523/JNEUROSCI.10-10-03227.1990
117. Krützfeldt NOE, Logerot P, Kubke MF, Wild JM (2010). Connections of the auditory brainstem in a songbird, *Taeniopygia guttata*. I. Projections of nucleus angularis and nucleus laminaris to the auditory torus. *Journal of Comparative Neurology* 518(11):2109-2134. https://doi.org/10.1002/cne.22334
118. Wild JM, Karten HJ, Frost BJ (1993). Connections of the auditory forebrain in the pigeon (*Columba livia*). *Journal of Comparative Neurology* 337(1):32-62. https://doi.org/10.1002/cne.903370103
119. Brauth SE, McHale CM, Brasher CA, Dooling RJ (1987). Auditory pathways in the budgerigar. I. Thalamo-telencephalic projections. *Brain, Behavior and Evolution* 30(3-4):174-199. https://doi.org/10.1159/000118645
120. Vates GE, Broome BM, Mello CV, Nottebohm F (1996). Auditory pathways of caudal telencephalon and their relation to the song system of adult male zebra finches. *Journal of Comparative Neurology* 366(4):613-642. https://doi.org/10.1002/(SICI)1096-9861(19960318)366:4%3C613::AID-CNE5%3E3.0.CO;2-7
121. Mello CV, Velho TA, Pinaud R (2004). Song-induced gene expression: a window on song auditory processing and perception. *Annals of the New York Academy of Sciences* 1016:263-281. https://doi.org/10.1196/annals.1298.021
122. Knudsen EI (2002). Instructed learning in the auditory localization pathway of the barn owl. *Nature* 417(6886):322-328. https://doi.org/10.1038/417322a
123. Hyde PS, Knudsen EI (2002). The optic tectum controls visually guided adaptive plasticity in the owl's auditory space map. *Nature* 415(6867):73-76. https://doi.org/10.1038/415073a
124. Heffner HE, Koay G, Heffner RS (2016). Budgerigars (*Melopsittacus undulatus*) do not hear infrasound: the audiogram from 8 Hz to 10 kHz. *Journal of Comparative Physiology A* 202(12):853-857. https://doi.org/10.1007/s00359-016-1125-9

**The parrot's vocal system**

125. Jarvis ED (2019). Evolution of vocal learning and spoken language. *Science* 366(6461):50-54. https://doi.org/10.1126/science.aax0287
126. Chakraborty M, Jarvis ED (2015). Brain evolution by brain pathway duplication. *Philosophical Transactions of the Royal Society B* 370(1684):20150056. https://doi.org/10.1098/rstb.2015.0056
127. Chakraborty M, Walløe S, Nedergaard S, Fridel EE, Dabelsteen T, Pakkenberg B, Bertelsen MF, Dorrestein GM, Brauth SE, Durand SE, Jarvis ED (2015). Core and shell song systems unique to the parrot brain. *PLoS ONE* 10(6):e0118496. https://doi.org/10.1371/journal.pone.0118496
128. Jarvis ED, Mello CV (2000). Molecular mapping of brain areas involved in parrot vocal communication. *Journal of Comparative Neurology* 419(1):1-31. https://doi.org/10.1002/(SICI)1096-9861(20000327)419:1%3C1::AID-CNE1%3E3.0.CO;2-M
129. Durand SE, Heaton JT, Amateau SK, Brauth SE (1997). Vocal control pathways through the anterior forebrain of a parrot (*Melopsittacus undulatus*). *Journal of Comparative Neurology* 377(2):179-206. https://doi.org/10.1002/(SICI)1096-9861(19970113)377:2%3C179::AID-CNE3%3E3.0.CO;2-0
130. Striedter GF (1994). The vocal control pathways in budgerigars differ from those in songbirds. *Journal of Comparative Neurology* 343(1):35-56. https://doi.org/10.1002/cne.903430104
131. Paton JA, Manogue KR, Nottebohm F (1981). Bilateral organization of the vocal control pathway in the budgerigar, *Melopsittacus undulatus*. *Journal of Neuroscience* 1(11):1279-1288. https://doi.org/10.1523/JNEUROSCI.01-11-01279.1981
132. Brauth SE, Heaton JT, Durand SE, Liang W, Hall WS (1994). Functional anatomy of forebrain auditory pathways in the budgerigar (*Melopsittacus undulatus*). *Brain, Behavior and Evolution* 44(4-5):210-233. https://doi.org/10.1159/000113578
133. Yang Z, Long MA (2025). Convergent vocal representations in parrot and human forebrain motor networks. *Nature* 640(8058):427-434. https://doi.org/10.1038/s41586-025-08695-8
