# Space: a map of what is where

A brain with a spatial model has a map of things. The map holds where things are, each place
bound to what the thing is. A thing enters the map when a sense reports it. Sensory information
updates its place. With no sensory information the last known place stays in the state, for any
length of time.

The module has two stages: **the map, then learned physics on the map.**

- **The map** (`cadence.space.map`, this guide) is a sheet of neurons whose settled activity is
  the whole state. No list of things is given to it and none is kept beside it.
- **Learned physics on the map** (the second stage, in work) lets a brain learn how things move
  from the things it watches moving on its map, and then infer how a thing it does not see has
  moved. The map is its control: a brain that has watched nothing answers as the map does.

The pieces are opt-in: a program that does not import `cadence.space` gets a library without it.

| Step | Piece | Section |
| --- | --- | --- |
| Make a map of a plane | `Map` | [a first map](#a-first-map) |
| Tell it what the senses report | `moment(seen=, heard=, felt=, lit=)` | [how a position travels](#how-a-position-travels) |
| Ask where a thing is, or what is at a place | `where_is`, `what_is_at` | [questions](#questions-answered-by-settling) |
| Read every thing it holds | `things`, `sheet` | [the inventory](#the-inventory) |
| A thing that leaves view | `empty`, `reach` | [a place seen empty](#a-place-seen-empty) |
| Keep the map | `save`, `Map.load`, `reset` | [the carried state](#the-carried-state) |
| The earlier form, kept as a baseline | `Space`, `Interface` | [the baseline](#the-baseline-places-held-by-a-program) |

## What the brain does and what the program does

| | The program beside the map | The brain (the map's graph) |
| --- | --- | --- |
| Where things are held between moments | Copies the sheet's potentials to the next moment's drive on the same neurons | In the settled activity of `map`, with `clock`, `doubt` and `lost` |
| The list of things | None: no cast, no rows | The peaks of the sheet |
| A report from a sense | Turns the kind and the place into a drive on four cells | The cells settle on it |
| Holding a place | Nothing beyond the copy | A neuron driven by its own last potential, with no synapse into it, settles at that potential: the place stays exactly |
| A place seen empty | Passes the cells in sight as the settle's mask; finds a piece of activity with no report of its kind and drives it onto the unseen cells within reach | A seen cell is held at zero in every sweep |
| Which report is which thing | Compares a report's place with the pieces of its kind (`step`, `stride`) | Two things of a kind are two peaks |
| Questions | Sets the cue: the kind's code drives the pools' references, or the mask keeps the cells around a place | `where` and `what` settle on the answer |
| The inventory | Reads the peaks, their cells and the clock | Holds them |
| Kinds | Takes each thing's kind from the caller as a number | Holds a kind as a code over the sheet's units |
| Lessons | None | None: the map's graph has no taught synapse |

The map is wiring and carried activity. Recognising a thing from pixels is another module's
work: the caller says "a thing of this kind at this place".

## A first map

```python
import numpy as np
from cadence.space.map import Map

APPLE, BREAD, HAND = 0, 1, 2
things = Map((-1.0, 1.0, -0.2, 1.4))                   # a plane of 2 m by 1.6 m
things.moment(seen=[(APPLE, [0.30, 0.62])])            # the camera reports an apple
things.moment(seen=[(BREAD, [-0.40, 1.00])])           # then a loaf: no thing is announced
for _ in range(100):
    things.moment()                                    # no sensory information
assert np.allclose(things.where_is(APPLE), [[0.30, 0.62]], atol=0.001)   # "where is the apple?"
assert things.what_is_at([-0.40, 1.00]) == BREAD       # "what is at this place?"
assert [t["kind"] for t in things.things()] == [APPLE, BREAD]
assert all(t["state"] == "held" and t["since"] >= 100 for t in things.things())
```

`Map(plane, cells=(32, 26), units=16)` makes the sheet: 32 by 26 cells over the plane and 16
units of a kind's code at each cell. A kind is a number. With 16 units the map holds 16 kinds,
one unit a kind. `Map(plane, units=32, kinds=128)` holds 128 kinds on 32 units: each kind takes
three units and two kinds share one unit at most. `codes=` takes a table of the caller's own.

## How a position travels

1. **A sense reports a thing.** A report is a kind and a place: the camera's "a hand at this
   place", the place a sound names, the place the body's own readings name. The caller gives the
   kind as a number.
2. **The report enters the map.** The place becomes weights on the four cells around it, whose
   mean is the place, and drives those cells on the units of the kind's code. A kind the map has
   not held enters the same way. No thing is announced beforehand.
3. **The map carries itself.** At the next moment the potentials the sheet settled to drive
   the same neurons, and a neuron with that drive and no synapse into it settles at that
   potential. With no report and no sight of the cell nothing changes, to the bit: the last
   known place is the place in the state.
4. **Sight again.** A report within `step` of a piece of its kind's activity is that thing. Its
   old cells are cleared and the report's cells are written, so in view a thing's place is the
   latest report.
5. **A place seen empty.** A cell the camera sees loses what the map carried into it. A thing
   whose place is seen empty, with no report of its kind near, is not where it was seen. The map
   drives it onto the cells out of sight within reach of that place and marks those cells as
   doubt. With no such cell the thing is kept by its kind with no place.
6. **It is found.** A report of the kind ends the doubt: the cells of doubt that hold the kind
   are cleared, and so is the nearest piece of the kind within `stride`, as the same thing moved
   unseen. With none of its kind that near, the report is a new thing.
7. **Questions.** "Where is the apple": the carried potentials drive `held`, the apple's code
   opens the pools of its units, the sheet settles on those units alone and `where` holds the
   apple's places. "What is there": the settle keeps the cells
   around the place and `what` holds the code of the thing there.
8. **The inventory.** Every piece of activity is a thing: its kind, its place (the mean of the
   piece), its cells, the moments since a sense wrote it, and its state.

```python
def camera_sees(points):                               # the caller knows its camera
    return points[:, 0] > -0.1                         # a shadow lies to the left of -0.1

room = Map((-1.0, 1.0, -0.2, 1.4))
room.moment(seen=[(HAND, [0.0, 0.6])], lit=camera_sees)
room.moment(lit=camera_sees)                           # the hand's place is seen empty
(hand,) = room.things()
assert hand["state"] == "near" and hand["place"][0] < -0.1       # out of sight, near there
assert not camera_sees(room.centers[hand["cells"]]).any()        # its cells: where it may be
room.moment(seen=[(HAND, [0.05, 0.6])], lit=camera_sees)         # it comes out
assert np.allclose(room.where_is(HAND), [[0.05, 0.6]], atol=0.001)
```

## The regions

Sizes for the founder, 32 by 26 cells and 16 code units: 29,168 neurons and 39,936 synapses.

| Region | Neurons | What it holds |
| --- | --- | --- |
| `map` | 13,312 | The sheet: a unit at a cell is active where a thing whose kind's code holds the unit is at the cell. At each moment its drive is its own potentials of the moment before |
| `held` | 13,312 | The question's input: for "where is" it is driven by the sheet's carried potentials and projects onto the sheet cell to cell |
| `reference` | 16 | One a code unit; a question's kind drives the references of its code |
| `where` | 832 | One a cell, reading `map` over the code units |
| `what` | 16 | One a code unit, reading `map` over the cells |
| `clock` | 832 | One a cell: the moments since a sense wrote the cell. It carries its own potential and a bias of one |
| `doubt` | 832 | One a cell: a one where the cell holds a thing that was not seen to be there |
| `lost` | 16 | One a code unit: a one where a thing of the kind is known with no place |

`held` projects onto `map` cell to cell, and `map` onto `where` and `what`, each synapse at an
efficacy of 2, the inverse of the neuron's slope at rest. Each code unit's cells and its
reference form one pool of an apical compartment ([the apical compartment](anatomy.md#the-apical-compartment)).
The pool's gain is the library's own, set inside the settle, and it is a multiplication: a pool
whose reference is silent passes nothing. The pools serve the question by kind alone; the carry
uses none of them.

**What was tried for the carry.** Two other forms were read before this one. With the sheet's
potentials driving `held` and the pools' gains holding each pool at the length its reference was
driven with, the program drove each reference with the length of what its pool carried in: a
place moved by 0.02 cm in 1,000 moments, from the neuron's curve. A reference neuron that sums
the carried activity in place of that number holds no place: the length a thing needs is the
root of its summed squares, which is its summed activity for a thing on a cell's middle and half
of it for a thing between four cells. The length held, in units of the length written, after
1, 5, 10 and 20 moments:

| Reference | A thing on a cell's middle | A thing between four cells |
| --- | --- | --- |
| A neuron summing the carried activity at a weight of 1 | 1, 1, 1, 1 | 2, 32, 368, 383 |
| The same at a weight of 0.5 | .25, .016, .0005, 5e-7 | 1, 1, 1, 1 |
| The length, set by the program | 1, 1, 1, 1 | 1, 1, 1, 1 |

The carry on the sheet's own neurons needs no gain and no set point, and it moves nothing.

## Questions, answered by settling

```python
kitchen = Map((-1.0, 1.0, -0.2, 1.4))
for kind, place in ((APPLE, [0.0, 0.0]), (APPLE, [0.7, 1.1]), (BREAD, [-0.6, 0.4])):
    kitchen.moment(seen=[(kind, place)])
apples = kitchen.where_is(APPLE)                       # two things of one kind: two places
assert len(apples) == 2 and kitchen.what_is_at([0.7, 1.1]) == APPLE
assert kitchen.what_is_at([0.9, -0.1]) is None         # nothing is held there
```

A question changes nothing in the map. `where_is(kind)` gives `(things, 2)` places, one a
connected piece of activity. `what_is_at(place, within=)` reads the cells within a cell and a
half of the place unless `within` says otherwise.

## The inventory

`things()` reads every thing off the map's carried activity, with no settle:

| Key | Value |
| --- | --- |
| `kind` | The kind's number |
| `place` | The mean of the thing's piece of activity, `(x, y)` in metres; `None` for a thing with no place |
| `cells` | The thing's cells, as numbers into `centers` (each cell is `size` metres around its middle) |
| `pieces` | The cells by connected piece. A thing in doubt that was driven onto two shadows is one entry with two pieces |
| `since` | The moments since a sense wrote it; 0 at the moment it is sensed |
| `state` | `"seen"`, `"heard"` or `"felt"` at the moment a sense writes it; `"held"` at its last known place; `"near"` on cells of doubt; `"not known where"` |

`sheet(kind)` gives the kind's activity as a picture, `(cells x, cells y)`, and `sheet()` every
unit summed.

`Map(..., trace=True)` keeps every settle by sweep in `trajectory`, for a reader who watches the
brain: `asked` (`"moment"`, `"where_is"` or `"what_is_at"`), `sweeps` and the activity of `map`,
`where` and `what` at each sweep. A watched settle begins with the sheet at rest and takes two
sweeps: the first settles the sheet on its drive, the second the two regions that read it.

## A place seen empty

A hand that goes into a shadow was last seen in a cell the camera sees, and at the next moment
that cell is seen empty. `empty` says what the map does then:

| `empty` | What becomes of a thing whose place is seen empty |
| --- | --- |
| `"near"` (the founder) | It is driven onto the cells out of sight around where it was held, as a peak `reach` metres wide; cells farther than two reaches take none. With no such cell it is kept by its kind with no place |
| `"clear"` | Its seen cells are cleared. A thing with no cell left is kept by its kind with no place |
| `"keep"` | It stays where the map held it: the last seen place, whatever the camera sees there |

A reading on the two recorded scenes of the chamber, in which a hand goes behind a pillar at
0.5 m/s, rests there for 70 moments and comes back
([`benchmarks/tracking/mapped.py`](../benchmarks/tracking/mapped.py)); the mean distance from
the hand over its 91 hidden moments:

| Arm | Shadows joined | Arm folded |
| --- | --- | --- |
| Last seen | 15.5 cm | 15.5 cm |
| The map, `empty="keep"` | 15.5 cm | 15.5 cm |
| The map, `empty="near"`, `reach=0.15` (the founder) | 11.8 cm | 11.8 cm |
| The map, `empty="near"`, `reach=0.3` | 11.3 cm | 11.2 cm |
| The map, `empty="near"`, `reach=0.6` | 16.3 cm | 17.2 cm |

In view the map reads 1.2 cm, the camera's own error. The founder's cells hold the hand at
every hidden moment, at 34 cells. The map does not move a thing along its last seen movement:
that is the second stage.

**How the reach was chosen.** The first value, 0.3 m, was fixed before any reading. It was
changed to 0.15 m on the development panels of one founder, before the declared run, and the two
recorded scenes above took no part in the choice. On 140 scenes of a person's hand and 40 scenes
of hands that visit a shadow, pause, hop, go slowly and are left, the error out of view:

| Arm | A person's hand | Hands that visit, pause, hop, go slowly, are left |
| --- | --- | --- |
| Last seen | 26.8 cm | 14.9 cm |
| The map, `empty="keep"` | 26.8 cm | 14.9 cm |
| The map, `reach=0.15` | 23.2 cm | 11.8 cm |
| The map, `reach=0.3` | 24.1 cm | 16.6 cm |
| The map, `reach=0.6` | 32.4 cm | 31.2 cm |

At 0.3 m a hand that pauses, hops or goes slowly near the shadow's edge is placed deep in the
shadow at once: in its first five moments unseen the map read 14.6 cm where last seen read
4.2 cm.

## The carried state

`arrays()` gives the state: the potentials of `map`, `clock`, `lost` and `doubt`. `save(path)`
writes the map's settings, its codes and that state as arrays and one JSON record, and
`Map.load(path)` reads them back. `reset()` empties the carried activity, and the map then
reports no thing.

```python
import tempfile
from pathlib import Path

path = kitchen.save(Path(tempfile.mkdtemp()) / "kitchen.npz")
again = Map.load(path)
assert np.allclose(again.where_is(APPLE), apples) and len(again.things()) == 3
again.reset()
again.moment()
assert again.things() == [] and len(again.where_is(APPLE)) == 0
```

## Settings and cost

| Setting | Founder | What it sets |
| --- | --- | --- |
| `cells` | `(32, 26)` | The cells an axis: 6 cm cells on a plane of 2 m by 1.6 m |
| `units`, `kinds`, `weight`, `codes` | 16, none, 3, none | The code units a cell, and the kinds' codes |
| `level` | 0.02 | The length a report's potentials are written at on each of its units |
| `empty`, `reach` | `"near"`, 0.15 m | What becomes of a thing whose place is seen empty |
| `step` | 0.15 m | A report within this distance of a piece of its kind is that thing, seen again |
| `stride` | 0.5 m | A report of a kind with none of it within `step`: the nearest piece within this distance is that thing, moved unseen |

One moment settles one row whatever the number of things. Under Pyodide the founder takes 1.9 ms
a moment with four things, 0.3 ms for the inventory and 1.8 ms a question; on one core natively
1.3 ms a moment and 1.6 ms a question.

## The baseline: places held by a program

`Space` is the earlier form: a program holds a place for every thing of a cast given at
`begin`, and a feed-forward brain answers how a place moves in a step. It stays in the library
as the baseline the map and the second stage are read against. Its sections follow as they were.

### What the baseline's brain does and what its program does

`Space` is a program beside a brain, as `cadence.LongTermMemory` is. The two share the work:

| | The program (`Space`) | The brain |
| --- | --- | --- |
| Where a place is held between moments | Two numbers a thing, with the radius of its area | Nowhere: the `place` region shows the program's place to the brain at each step |
| A report from the camera | Writes the thing's place; a thing at rest takes the mean of its reports | Reads nothing of it at that step |
| A thing's seen movement | The slope of the line through its reports in a row | Reads it in the `motion` region |
| How far a place moves in a step unseen | Adds the answer to the place | Answers it by settling: `shift` |
| How far off a place may be | Reads the distance that holds .90 of the belief and reports a circle | Answers it by settling: `spread` |
| The place a cue or a posture names | Writes the answer as the place | Answers it by settling: `found` |
| Which report belongs to which thing | Gives each report to the nearest row of its look | Takes no part |
| Things at rest, by what they look like | Files and revises a record in a `cadence.LongTermMemory` | Takes no part |
| Lessons | Makes each lesson's target from the camera's reports | Learns by the free and nudged phases of [the library's rule](settling.md#learning-the-freenudged-rule) |

So the peak a reader sees in the `place` region is the program's place, shown to the brain. The
brain's own states are the hidden region and the three answers.

### A first baseline space

```python
import numpy as np
from cadence.space import Interface, Space

interface = Interface(plane=(-1.0, 1.0, -0.2, 1.4), hidden=32)   # a plane of 2 m by 1.6 m
space = Space(interface, seed=0, backend="cpu")

space.begin(np.array([[0.2, 0.5], [-0.4, 1.0]]))     # two things at their starting positions
seen = np.array([True, False])                       # the camera reports the first one
reports = np.array([[0.30, 0.62], [0.0, 0.0]])       # at another place than it started
places = space.advance(seen, reports)                # one step
where, radius = space.where()                        # each place, and its area's radius at .90
assert np.allclose(where[0], [0.30, 0.62]) and np.allclose(where[1], [-0.4, 1.0])
assert radius.shape == (2,)
```

`advance` takes one step for every thing (a row each): which rows the camera reports, the
reports, and where they apply the arm's command, the cues and the joint readings. A row with no
report, no command, no seen movement and no cue is left as it is, and where no row has any the
brain is not asked:

```python
nothing = np.zeros(2, dtype=bool)
for _ in range(1000):
    space.advance(nothing, np.zeros((2, 2)))
assert np.allclose(space.where()[0], where) and space.work["moments"] == 0
```

### Lessons hold only what the camera reports

With `learn=True` every step also teaches the brain, from the reports alone:

- **`shift`** is taught the movement the camera reports: the slope of six reports in a row. For
  an arm's own end the six lie around the step taught. For a thing that moves by itself they are
  the six from the step taught on, so the lesson shares no report with what the answer read.
- **`spread`** is taught how far a predicted place lay from the report that followed it.
- **`found`** is taught the reported place at the moments a thing is heard, or the arm's joints
  are read, while it is seen.
- **Practice.** At times the space leaves a thing's reports unread for a span of steps
  (`practice`, the rate a span begins; `span`, its mean length). Over the span it moves the place
  as if the thing were out of view, and the reports it did not read teach the answers it gave.

No lesson holds a position the camera did not give.

```python
rng = np.random.default_rng(0)
start = np.array([[-0.6, 0.2], [0.5, 0.3], [0.0, 0.8], [-0.2, 1.0]])
velocity = np.array([[0.010, 0.004], [-0.008, 0.006], [0.0, 0.0], [0.006, -0.010]])
space.begin(start)
for t in range(1, 40):
    truth = start + t * velocity
    space.advance(np.ones(4, dtype=bool), truth + rng.normal(0.0, 0.01, (4, 2)), learn=True)
assert space.work["lesson_row_sweeps"] > 0
assert 0.005 < space.state()["noise"] < 0.02          # the camera's noise, learned from reports
```

Two readings of the brain's `shift` and one control (`read=`, which can be set at any step):

| `read` | What moves a thing that moves by itself while it is unseen |
| --- | --- |
| `"mean"` | The mean of the brain's belief. A brain that saw things slow, stop and turn while unseen believes in all of them, and moves the place part of the way. |
| `"peak"` | The value at the belief's highest peak: the likeliest movement. |
| `"seen"` | The control: the thing's last seen movement as it is, whatever the brain answers. |

### The baseline's regions

`Interface` is the record of the brain's regions and what their units stand for, and
`cadence.space.compose(interface)` composes the brain. Sizes for an arm of two joints:

| Region | Units | What it holds |
| --- | --- | --- |
| `place` | 32 | The held place: a peak on each axis, 16 tuned positions an axis, evenly spaced over the plane. Written by the program. |
| `moved` | 18 | The copy of the arm's command: a peak a joint. Dark for a thing that is no arm's end and for a command of zero. |
| `motion` | 56 | The thing's last seen movement, a peak an axis over 25 tuned speeds, and how many reports it was seen over. Dark until three reports in a row. |
| `unseen` | 8 | The moments since the thing was seen. |
| `heard` | 320 | A cue from the ear: one unit a pair of direction and distance, a sheet of 20 by 16. Dark while silent. |
| `felt` | 256 | The arm's joint readings: one unit a posture, 16 readings a joint. Dark where no readings are given. |
| `hidden` | 128 | Reads every input. |
| `shift` | 64 | How far the place moves in this step: 32 tuned movements an axis, from -6 cm to 6 cm. |
| `spread` | 16 | How far off a place predicted here is found to be: 16 tuned distances from 2 mm to 2 m. |
| `found` | 48 | The place a cue or a posture names: 24 tuned positions an axis. |
| `keep` | 8 | The share of its seen movement a thing that moves by itself keeps in this step, from 0 to 1 (read with `course=True`). |

The inputs drive `hidden`, `hidden` drives the outputs, and one-way synapses that start at zero from `motion` and `moved` to `shift`, from
`unseen` and `motion` to `spread` and to `keep`, and from `heard` and `felt` to `found`. With
`Interface(feedback=True)` the outputs also drive the hidden region, and the brain settles
between the two. Without it a settle has one answer whatever it began from: that arm is the
baseline the reciprocal brain is read against.
Each output is one softmax, read at a temperature of 0.05.

```python
assert list(interface.regions) == ["place", "moved", "motion", "unseen", "heard", "felt",
                                   "hidden", "shift", "spread", "found", "keep"]
state = space.state()                                  # plain arrays, one row a thing
assert state["place"].shape == (4, interface.regions["place"])
assert state["at"].shape == (4, 2) and state["radius"].shape == (4,)
```

`state()` gives, after the latest step: `at` (each place), `radius`, `since` (the moments since
each thing was seen), `noise` (the camera's learned noise), each input region's activity by its
name, `live` (the rows the brain settled on) and for those rows `hidden` and the beliefs
`shift`, `spread` and `found`. `Space(track=True)` keeps the synapses the latest lesson moved in
`changed` and how far in `changes`.

### How a position travels in the baseline

1. **The camera's report.** A report is a position in the plane. The program writes it into the
   thing's place. The `place` region shows that place to the brain as a peak on each axis: the
   units within two spacings of the position are lit, five at most, and the rest are silent. A
   thing that is not seen changes nothing here: there is no "nothing seen" pattern.
2. **The place is held.** The program keeps the place between moments. With no report, no
   command, no seen movement and no cue it is not touched, and the brain is not asked. A thing
   at rest in view takes the mean of its reports, which is sharper than one report.
3. **Movement while unseen.** The brain settles on what the regions hold and its `shift`
   answers how far the place moves in this step. For the arm's own end the `moved` region holds
   the copy of the command. For a thing that moves by itself the `motion` region holds its last
   seen movement: the slope of the line through its reports in a row. The program adds the shift
   to the place.
4. **Correction by sight.** When the thing is seen again its report writes the place. The
   lessons happen only then: the movement the reports show teaches `shift`, and the distance
   from the predicted place to the report teaches `spread`.
5. **The uncertainty.** `spread` is a belief over distances. The program reads the distance
   that holds .90 of the belief, takes out the camera's own noise, and reports a circle of that
   radius around the place. The circle grows while the thing is unseen and shrinks to the
   camera's noise when it is seen.
6. **Things that stay put.** With `records`, a thing with a look of its own that rests is filed
   in a `cadence.LongTermMemory`: the key is a code of what it looks like and the value is its
   place and the number of reports the place is a mean of. The record is revised as the mean
   sharpens and when the thing is seen elsewhere.
7. **A cue from another sense.** A cue is a direction and a distance from the ear. It lights
   one neighbourhood of the `heard` sheet, the brain's `found` answers the place it names, and
   the program writes that place. The same holds for the arm's joint readings in `felt`.
8. **The answer.** `where()` gives each place and its radius. `where_is(kind)` answers a kind
   with its places, and `what_is_at(point)` answers a place with the kind held there.

### Several things

A camera's report carries a position and what the thing looks like, and no name. `begin` takes
each row's look (its kind, which several things may share). `assign` gives the reports of a step
to rows of their look, and `hear` gives the cues to rows:

```python
room = Space(interface, seed=1, backend="cpu")
room.begin(np.array([[0.0, 0.0], [0.5, 0.0], [0.0, 1.0]]), looks=[3, 3, 7])   # two apples, a loaf
cues = room.hear(np.array([[-2.2, 1.4, 7]]))           # the ear: direction, distance, look
room.predict(cues=cues)                                # the first half of a step
seen, reports = room.assign(np.array([[0.52, 0.01, 3], [0.01, 0.02, 3]]))   # x, y, look
room.observe(seen, reports)                            # the second half
rows, places, radius = room.where_is(3)                # "where are the apples?"
near = np.linalg.norm(places - [[0.0, 0.0], [0.5, 0.0]], axis=1)
assert rows.tolist() == [0, 1] and near.max() < 0.03   # each report went to the nearer apple
assert room.what_is_at([0.5, 0.0]) == (1, 3)           # "what is at this place?"
```

`advance` is `predict` and then `observe`. Reports of one look go to the rows of that look at
the least summed distance from where each row is if it kept its last seen movement, so two hands
that cross out of view are told apart when they come back.

### Records of things at rest

```python
keeper = Space(interface, seed=2, backend="cpu", records=64)
keeper.begin(np.array([[0.2, 0.3]]), looks=[4])
for _ in range(30):
    keeper.advance([True], np.array([[0.2, 0.3]]) + rng.normal(0.0, 0.01, (1, 2)))
place, count = keeper.find(4)                          # the record of look 4
assert np.linalg.norm(place - [0.2, 0.3]) < 0.01 and count == 16
keeper.begin(np.array([[np.nan, np.nan]]), looks=[4])  # a new scene: the record gives its place
assert np.linalg.norm(keeper.where()[0][0] - [0.2, 0.3]) < 0.01
```

### A trained module

`space.save(path)` writes the trained regions as a module file (`cadence.modules`: arrays and
one JSON record, with a checksum for every array), with the interface and the camera's learned
noise in its record. `Space.load(module)` makes a space of the module's own
brain. `Brain.compose(..., trained=[module])` builds a brain around the module, and
`Space.load(module, brain=host)` runs the space on the regions inside that brain:

```python
import cadence as cd
from cadence import Brain

path = space.save("place.npz", card={"trained on": "forty steps of this page"})
module = cd.load_module(path)
host = Brain.compose(inputs=5, actions=3, seed=1, trained=[module], backend="cpu")
hosted = Space.load(module, brain=host)                # the place regions inside the host
alone = Space.load(module)
hosted.begin(np.array([[0.2, 0.6]]), own=True)
alone.begin(np.array([[0.2, 0.6]]), own=True)
command = np.array([0.04, -0.03])
a = hosted.predict(command, beside=np.zeros((1, 5)))   # the host's own five inputs beside
b = alone.predict(command)
assert np.allclose(a, b)                               # the skill arrives as it was
```

In a host the module's inputs follow the host's own in the observation, its regions are the
populations `place/...`, its synapses are held, and the host's first module region reads the
held place and the four answers (`shift`, `spread`, `found`, `keep`) through one-way synapses that
start at zero. No synapse leads from the host into the module, so it answers in a host as it does
alone. A space in a host takes no lessons (`predict`, `observe` and `advance` refuse
`learn=True`): the regions were taught in their donor. `advance` and `predict` take the host's
other inputs as `beside=`.

### Settings, refusals and cost

| Setting | Default | What it sets |
| --- | --- | --- |
| `practice`, `span` | 0.04, 30 | The rate a span of unread reports begins at a step in view, and its mean length |
| `read` | `"mean"` | How `shift` is read: `"mean"`, `"peak"`, or `"seen"`, the control |
| `smooth` | `True` | A thing in steady movement takes, in view, the place its line gives for the latest report; `False` takes the report itself |
| `sight` | `None` | For a camera and a blind spot that stay where they are: a place out of view does not move into a cell where things are seen in more than this share of the visits |
| `records` | `0` | Records of things at rest, by what they look like |
| `course` | `False` | A thing that moves by itself is moved by its seen movement times the brain's `keep`, in place of the brain's `shift` |
| `patience` | `3` | With `sight`, or with flags for `in_sight`: the steps in a row a place is predicted in sight with no report before it is placed halfway between where it was last seen and where it was before those steps |
| `cells` | `32` | The cells an axis of the map of where things were seen (`sight`) |
| `track` | `False` | Keep the synapses each lesson moved |
| `module` | `None` | The name of the place regions inside a host brain |

`Interface` refuses a plane whose low is not below its high, a region of under two units, and a
`felt` sheet of more than 4,096 postures. A space in a host refuses lessons. The brain of the
default interface holds 140,731 parameters; one step on one row takes under a millisecond on one
core.

**Sight that does not come.** `observe(seen, reports, in_sight=f)` takes a function of
`(points, 2)` places that says, one flag a place, whether the camera would see a thing there at
this step; the caller knows the camera and what stands before it. A thing that goes out of view
while moving is then looked ahead for along its seen movement: its hidden stretch ends where it
would come back into view, or at the plane's edge. It is carried no farther than the middle of
that stretch and waits there, and its radius reaches .90 of the way back to where it was last
seen. `in_sight` also takes one flag a row for the places `predict` gave; `sight=` learns the
same from where things were seen, for a camera and a blind spot that stay where they are. A thing
at rest whose own place the caller says is in sight, with no report from there for `patience`
steps, is marked gone (`state()["gone"]`) and its radius becomes the plane's until it is seen.
Where the hidden stretch has no lit end (one shadow running into another), no such evidence
arrives and the place follows the brain's `keep` alone.

A module file made by another version of the place brain is refused by `Space.load` with the
version that made it: make the file again with the version that loads it.
