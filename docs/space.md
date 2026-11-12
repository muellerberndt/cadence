# Space: where things are while they are out of view

This guide gives a program a map of where things are. A camera reports a thing while it sees
it. The map keeps the thing's place while the camera does not: it leaves a thing at rest where it
was, moves a thing that was moving, takes a cue from another sense, and says how far off each
place may be.

The pieces are in `cadence.space`, an opt-in package: a program that does not import it gets a
library without it.

| Step | Piece | Section |
| --- | --- | --- |
| Name the plane and the arm | `Interface` | [the regions](#the-regions) |
| Hold the places of things | `Space`, `begin`, `advance`, `where` | [a first map](#a-first-map) |
| Teach it from the camera's reports | `advance(..., learn=True)` | [lessons](#lessons-hold-only-what-the-camera-reports) |
| Reports without names, cues, kinds | `assign`, `hear`, `where_is`, `what_is_at` | [several things](#several-things) |
| Things that stay put | `records`, `find` | [records](#records-of-things-at-rest) |
| Save it, and build a brain around it | `save`, `Space.load`, `Brain.compose(..., trained=)` | [a trained module](#a-trained-module) |

## What the brain does and what the program does

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

## A first map

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

## Lessons hold only what the camera reports

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

## The regions

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

## How a position travels

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

## Several things

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

## Records of things at rest

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

## A trained module

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
held place and the three answers through one-way synapses that start at zero. A space in a host
takes no lessons: the regions were taught in their donor.

## Settings, refusals and cost

| Setting | Default | What it sets |
| --- | --- | --- |
| `practice`, `span` | 0.04, 30 | The rate a span of unread reports begins at a step in view, and its mean length |
| `read` | `"mean"` | How `shift` is read: `"mean"`, `"peak"`, or `"seen"`, the control |
| `smooth` | `True` | A thing in steady movement takes, in view, the place its line gives for the latest report; `False` takes the report itself |
| `sight` | `None` | For a camera and a blind spot that stay where they are: a place out of view does not move into a cell where things are seen in more than this share of the visits |
| `records` | `0` | Records of things at rest, by what they look like |
| `course` | `False` | A thing that moves by itself is moved by its seen movement times the brain's `keep`, in place of the brain's `shift` |
| `patience` | `3` | With `sight`: the steps in a row a place is predicted in sight before it goes back and rests |
| `track` | `False` | Keep the synapses each lesson moved |
| `module` | `None` | The name of the place regions inside a host brain |

`Interface` refuses a plane whose low is not below its high, a region of under two units, and a
`felt` sheet of more than 4,096 postures. A space in a host refuses lessons. The brain of the
default interface holds about 139,000 parameters; one step on one row takes under a millisecond
on one core.
