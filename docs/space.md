# Space: a map of what is where

A brain with a spatial model has a map of things. The map holds where things are, each place
bound to what the thing is. A thing enters the map when a sense reports it. Sensory information
updates its place. With no sensory information the last known place stays in the state, for any
length of time.

The module has two stages: **the map, then learned physics on the map.**

- **The map** (`cadence.space.map`, this guide) is a sheet of neurons whose settled activity is
  the whole state. No list of things is given to it and none is kept beside it.
- **Learned physics on the map** (`cadence.space.physics`, opt-in with `Map(physics=True)`):
  the map predicts where a thing it cannot see has gone, by how it has seen things move
  ([below](#learned-physics-on-the-map)).

The pieces are opt-in: a program that does not import `cadence.space` gets a library without it.

| Step | Piece | Section |
| --- | --- | --- |
| Make a map of a plane | `Map` | [a first map](#a-first-map) |
| Tell it what the senses report | `moment(seen=, heard=, felt=, lit=)` | [how a position travels](#how-a-position-travels) |
| Ask where a thing is, or what is at a place | `where_is`, `what_is_at` | [questions](#questions-answered-by-settling) |
| Read every thing it holds | `things`, `sheet` | [the inventory](#the-inventory) |
| A thing that leaves view | `empty`, `reach` | [a place seen empty](#a-place-seen-empty) |
| Keep the map | `save`, `Map.load`, `reset` | [the carried state](#the-carried-state) |
| Hold it in a brain, or load the built-in one | `Brain.space`, `cadence.modules.load("place")` | [a brain that holds a map](#a-brain-that-holds-a-map) |
| The arm's own hand and sounds | `felt=`, `heard=`, `Sense` | [other senses](#other-senses) |
| Predict where an unseen thing has gone | `Map(physics=True)`, `forecast`, `Physics` | [learned physics](#learned-physics-on-the-map) |
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
5. **A place seen empty.** The camera sees a thing's held place (the mean of its activity) and
   no report of its kind claims it: the thing is not where it was held. The map drives it, once,
   onto the cells out of sight within reach of that place and marks those cells as doubt. With
   no such cell the thing is kept by its kind with no place. A place reported the moment before
   is a report, as far off as the camera's error: it is seen empty where the camera sees all it
   saw around it (within `margin`) when it reported it. A camera turning off a thing leaves it
   held; a hand that walks into a shadow, which a camera that did not move sees gone, is placed
   out of sight near there. Any other held place is seen empty only where the camera sees all
   of it within `margin`, so the camera's error never decides it.
6. **Sight only takes away.** A thing in doubt is never moved again by sight: the cells of its
   patch that the camera sees are taken away, and with none left it is kept by its kind with
   no place.
7. **It is found.** A report of the kind ends the doubt: the cells of doubt that hold the kind
   are cleared, and so is the nearest piece of the kind within `stride`, as the same thing moved
   unseen. With none of its kind that near, the report is a new thing.
8. **Questions.** "Where is the apple": the carried potentials drive `held`, the apple's code
   opens the pools of its units, the sheet settles on those units alone and `where` holds the
   apple's places. "What is there": the settle keeps the cells
   around the place and `what` holds the code of the thing there.
9. **The inventory.** Every piece of activity is a thing: its kind, its place (the mean of the
   piece), its cells, the moments since a sense wrote it, and its state. The pieces a thing in
   doubt or in motion left, sensed last at one moment, are one thing.

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
place moved by 0.02 cm in 1,000 moments, from the neuron's curve (read at 6d46a5d6). A
reference neuron that sums the carried activity in place of that number holds no place: the
length a thing needs is the root of its summed squares, which is its summed activity for a thing
on a cell's middle and half of it for a thing between four cells. The length held, in units of the length written, after
1, 5, 10 and 20 moments (read at 880299c2):

| Reference | A thing on a cell's middle | A thing between four cells |
| --- | --- | --- |
| A neuron summing the carried activity at a weight of 1 | 1, 1, 1, 1 | 2, 32, 368, 383 |
| The same at a weight of 0.5 | .25, .016, .0005, 5e-7 | 1, 1, 1, 1 |
| The length, set by the program | 1, 1, 1, 1 | 1, 1, 1, 1 |

The carry on the sheet's own neurons needs no gain and no set point, and it moves nothing:
eight things, two of one kind, read within 0.0035 cm after 1,000 moments with nothing sensed,
their potentials unchanged to the bit (read at 6c8a352d).

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

`area(kind, rate=0.9)` gives the smallest set of cells that holds that share of a kind's
activity, as flags on the cells.

`Map(..., trace=True)` keeps every settle by sweep in `trajectory`, for a reader who watches the
brain: `asked` (`"moment"`, `"where_is"` or `"what_is_at"`), `sweeps` and every region's
activity at each sweep (`map`, `held`, `where`, `what`, `reference`, `clock`, `doubt`, `lost`). A
watched settle begins with the sheet at rest and takes two sweeps: the first settles the sheet
on its drive, the second the two regions that read it.

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
the hand over its 91 hidden moments, read at 829e8419:

| Arm | Shadows joined | Arm folded |
| --- | --- | --- |
| Last seen | 15.5 cm | 15.5 cm |
| The map, `empty="keep"` | 15.5 cm | 15.5 cm |
| The map, `empty="near"`, `reach=0.15` (the founder) | 10.7 cm | 10.7 cm |
| The map, `empty="near"`, `reach=0.3` | 12.3 cm | 11.2 cm |
| The map, `empty="near"`, `reach=0.6` | 20.4 cm | 19.6 cm |

In view the map reads 1.2 cm, the camera's own error. The founder's smallest set of cells
holding .90 of the hand's activity holds the hand at every hidden moment, at 30 cells. The map does not move a thing along its last seen movement:
that is the second stage.

**How the reach was chosen.** The first value, 0.3 m, was fixed before any reading. It was
changed to 0.15 m at 880299c2, on the development panels of one founder, before the declared run,
and the two recorded scenes above took no part in the choice. On 140 scenes of a person's hand
and 40 scenes of hands that visit a shadow, pause, hop, go slowly and are left, the error out of
view, read again at 829e8419:

| Arm | A person's hand | Hands that visit, pause, hop, go slowly, are left |
| --- | --- | --- |
| Last seen | 26.8 cm | 14.9 cm |
| The map, `empty="keep"` | 26.8 cm | 14.9 cm |
| The map, `reach=0.15` | 23.2 cm | 12.2 cm |
| The map, `reach=0.3` | 24.8 cm | 17.5 cm |
| The map, `reach=0.6` | 33.7 cm | 32.4 cm |

At 0.3 m a hand that pauses, hops or goes slowly near the shadow's edge is placed deep in the
shadow at once: in its first five moments unseen the map reads 15.0 cm where last seen reads
4.2 cm.

## The carried state

`arrays()` gives the state: the potentials of `map`, `clock`, `lost` and `doubt`.
`save(path, card=)` writes the map's settings, its codes and that state as arrays and one JSON
record, with a checksum of every
array and a card. `Map.load(path)` reads them back, and refuses by name a file with a pickled
object, an array that differs from its checksum, or arrays that do not fit the map's cells.
`reset()` empties the carried activity, and the map then reports no thing.

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

## A brain that holds a map

A brain holds a map as a part of itself: `brain.space = Map(...)`. `Brain.save` writes the map
with the brain and `Brain.load` restores it, its carried state included. The map's regions are a
graph of their own inside the brain object, beside the brain's connectome and outside it. The
brain reads the map through `observation()`: the activity of `where` (one value a cell) and of
`what` (one a code unit) as the latest moment settled them.

```python
import cadence as cd

held = cd.modules.load("place")                        # the map the library ships
host = cd.Brain.compose(inputs=held.count + held.units, actions=2, seed=0, backend="cpu")
host.space = held
host.space.moment(seen=[(APPLE, [0.2, 0.5])])
host.act(host.space.observation()[None])               # the brain reads where and what
again = cd.Brain.load(host.save(Path(tempfile.mkdtemp()) / "host.npz"))
assert again.space.what_is_at([0.2, 0.5]) == APPLE
```

`cadence.modules.load("place")` gives the built-in map, and `cadence.load_module(path)` a map
file from outside the library; both are a `Map`.

## Other senses

A sound's direction and distance, and the arm's own joint readings, name a place; the caller
passes the place: `moment(heard=[(kind, place)])` and `moment(felt=[(kind, place)])`. The place
the arm's joint readings name is the arm's forward kinematics, which the program computes as a
robot knows its own arm. On 40 of the chamber's scenes of the arm's own hand (founder 0, read
at 829e8419), the map fed so reads 0.5 cm while the hand is out of view, where the true
kinematics read 0.1 cm and last seen 12.6 cm.

**What was tried.** `cadence.space.map.Sense` is a taught projection of a reading onto the map's
cells: a sheet of tuned readings, one synapse a unit and cell, taught by the library's free and
nudged phases with a quadratic nudge at the moments the thing is felt while seen. Taught on 300
scenes of the arm's own hand, the place it names read 12.0 cm out of view at 12 tuned readings a
joint and 8.8 cm at 24 (read at 829e8419): the tuned sheet is too coarse for the arm's forward
map at this size.

## Learned physics on the map

Without physics the map holds a thing it does not see where it was last. With physics it also
predicts where the thing has gone, by how it has seen things move. It is one switch:

```python
held = Map((-1.0, 1.0, -0.2, 1.4), physics=True)  # learns while it watches
held.moment(seen=[(HAND, (0.1, 0.5))], lit=camera_sees)  # as before
held.things()                 # a hidden thing that moves: state "moving", with its "velocity"
ahead = held.forecast(10)     # the map 10 moments on, a copy: nothing is learned from it
ahead.where_is(HAND)          # where the hand will be
held.physics.describe()       # what the brain has learned
```

A fresh physics has learned nothing and the map answers exactly as it does without physics. It
learns while things move in view, and from then on predicts.

### What it learns

Four numbers, and nothing else about movement is given:

| Number | What it says |
| --- | --- |
| `rest` | The share of moving things that come to rest in a moment |
| `leave` | The share that leave their line in a moment: they slow or turn |
| `ahead` | How far along its old line a thing that left goes, as a share of its speed |
| `aside` | How fast it strays from where that puts it, as a share of its speed |

The rule is the same at every place. The plane's edge is the map's own boundary: a thing does
not leave the plane.

### What the brain sees of motion

While a thing is in view its reports are a run, and a line through the latest of them gives its
place and its velocity. Each report's surprise (its distance from where the line put it, against
the line's own error) corrects both. The line knows its own error exactly: of its place, of its
velocity and of the two together, carried through every correction. A surprise beyond three
standard deviations at two moments in a row ends the line, and it begins anew at the latest
reports: the thing has changed its motion. Whether a thing moves at all is the evidence of its
line against a thing at rest.

### How it learns: imagined lessons

Every 8 moments of a line that fits, the brain imagines the thing out of view from there, and
for the next 30 moments compares where the imagined line puts it with where it is seen: how far
short of its line it fell, and how far aside it went. Each lesson is explained by the outcomes
that fit it (it kept its line; it came to rest at moment j; it left its line at moment j),
weighted by how likely each is, and the counts of those outcomes give the four numbers (the
expectation-maximisation step, online). A line that would leave the plane teaches nothing. A
fresh brain counts 500 moments of things at rest before its first lesson, so it begins as the
map and moves away from it as far as its lessons carry it.

### Out of view: core, trail and tail

A thing that goes out of view while moving is held in three parts, all on the map's own sheet:

- **the core**, the things that kept their line: a narrow bump that moves by the thing's
  velocity, as wide as the line's own error;
- **the trail**, the things that came to rest: still activity where they stopped;
- **the tail**, the things that left their line: a bump that goes `ahead` from where they left
  and widens `aside`.

Each moment a share `rest` of the core and of the tail comes to rest where it is, a share
`leave` of the core joins the tail, and the core moves on (at the plane's edge it rests). Sight
takes away what it sees: the cells of the trail it sees, and of each bump the share the camera
sees, the bump then centred on what is left. The parts keep their sum, so what sight takes from
one part goes to the others: a hand that does not come out where its line says has stopped or
turned, and the map holds it there. The thing's whole pattern over the units moves with it.

The cells that hold a thing also hold what the brain knows of its motion (its line, its lessons
in progress and, out of view, its core and tail): the map keeps no list of things beside it.

### What the brain does and what the program does

| | The program beside the map | The brain (the map's graph) |
| --- | --- | --- |
| The line through a thing's reports | Corrects it by each report's surprise; ends it on a surprise beyond its error | Carries it at the thing's cells (`motion`) |
| A thing out of view that moves | Moves its core and tail, leaves the shares at rest on the trail, takes away what sight sees | Holds the three parts as settled activity of `map` |
| Lessons | Imagines the line from earlier moments and counts the outcomes that explain where the thing was seen (expectation-maximisation) | None: the four numbers are no synapses |

The physics learns by a counting rule in the program, driven by what it sees go differently
from what it imagined. It is no local repair inside the map's settle.

### Readings

READINGS

### Settings

`Map(physics=True)` takes the founder; `physics={...}` gives settings by name, and a `Physics`
may be shared by several maps (one of them taking the lessons: `learning=False` on the others).

| Setting | Founder | What it sets |
| --- | --- | --- |
| `noise` | 0.01 m | The camera's error, an axis |
| `line` | 30 | The reports a line runs through |
| `surprise` | 3.0 | Standard deviations of a surprise that end a line (two in a row) |
| `settled` | 1.5 | The mean squared surprise under which a line fits and may teach |
| `every`, `horizon` | 8, 30 | A lesson begins every 8 moments and runs for 30 |
| `reports` | 10 | The reports a line needs before it teaches |
| `prior` | 500 | Moments of things at rest counted before any lesson |
| `learning` | True | Whether this map's physics takes lessons |

### Limits

- **Changes are instant in the model.** A hand in the chamber eases to rest or turns over 8
  to 10 moments; the lessons read such a change as more than one, and the numbers read about
  twice the world's own rate of change. A share that goes straight ahead with almost no spread
  can sit in `leave` where nothing leaves its line: the lessons cannot tell it from keeping the
  line.
- **The same rule everywhere.** Where things come to rest (a table, a wall inside the room) is
  not learned; the plane's edge is the only boundary.
- **One thing, one record.** Two things of a kind that move out of view near each other are
  read as the map reads them (`step`, `stride`); their crossing is not read.

## Settings and cost

| Setting | Founder | What it sets |
| --- | --- | --- |
| `cells` | `(32, 26)` | The cells an axis: 6 cm cells on a plane of 2 m by 1.6 m |
| `units`, `kinds`, `weight`, `codes` | 16, none, 3, none | The code units a cell, and the kinds' codes |
| `level` | 0.02 | The length a report's potentials are written at on each of its units |
| `empty`, `reach` | `"near"`, 0.15 m | What becomes of a thing whose place is seen empty |
| `step` | 0.15 m | A report within this distance of a piece of its kind is that thing, seen again |
| `stride` | 0.5 m | A report of a kind with none of it within `step`: the nearest piece within this distance is that thing, moved unseen |
| `margin` | 0.02 m | The camera's error: the distance around a held place over which sight decides whether it is seen empty |

One moment settles one row whatever the number of things. At 6c8a352d, on a Mac under a load of
30 to 57 from other work, the founder takes 9.5 ms a moment with four things under Pyodide,
1.5 ms for the inventory and 7.1 ms a question; on one core natively 5.3 ms a moment and
6.2 ms a question.

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
