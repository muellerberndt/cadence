# Assemble a brain

This guide puts a brain together from its parts: senses, a map of things, memories, attention, a
voice. Read [start here](start.md) first; it takes one small brain through a life. Every example
on this page runs as written, in order.

A Cadence brain is one network of neurons that settles. At every moment its neurons drive each
other until their activity comes to rest; the action is read from that resting state, and every
synapse learns by one rule from the difference between two such settles. Binding comes with
this: everything wired into one brain settles into one joint state, so a sound heard while a
thing is in view is part of that moment as the thing's colour is. No part does the binding. What
you attach are **senses** (what reaches the brain), **stores** (what it keeps for later: a map of
where things are, records of what its senses held together) and **bodies** (what it does:
actions, a voice). `Brain.compose(inputs=4, actions=2)` with nothing else named is a whole brain;
every part below is added to it.

## What to use for which need

| You need | Part | The line |
| --- | --- | --- |
| A brain that acts and learns from what the world pays | the default brain | `Brain.compose(inputs=4, actions=2, arousal=True)` |
| Several senses, each given by name | [senses by name](#senses) | `senses={"eyes": 32, "ear": 80}` |
| A seen thing's colour and form as inputs | a sense for things | `Things({"hue": (0.0, 1.0)})` |
| A heard word as one code | an ear | `Ear(frames=16, bands=40)` |
| Senses that count alike, whatever their width | equal senses | `code=512` beside `senses=` |
| A body that acts on where things are from it (a bearing, a distance), its memories keyed on the code | [the senses beside the code](#a-body-that-acts-on-what-it-senses) | `code=128, cortex_reads="both"` |
| An action with parts, or one choice per motor | [motor slots](#actions-and-a-voice) | `slots=(3, 2)` |
| Where things are, in view and out of it | [the map of things](#the-map-of-things) | `trained=["place"]` |
| A thing's colour kept with it on the map while it is hidden | values on the map | `trained=[Map(plane, values={"hue": 8})]` |
| The one thing at a place, read whole | `thing_at` | `room.thing_at([0.3, 0.6])` |
| A hidden thing that moves, followed | learned physics on the map | `Map(plane, physics=True)` |
| Recent moments, back by their content | [the recall memory](#memory) | in every new brain |
| What paid in a situation | the associative store | in every new brain |
| A memory kept when it comes again or comes with a surprise | a memory's strength | `recall_strength=True` |
| A signal for what a moment brought that the brain did not hold | [surprise](#attention) | `surprise=True`, `brain.surprise` |
| One place attended, drawn by surprise | the focus | `brain.focus` |
| A sound heard alone that brings its thing back | [records of bound moments](#symbols) | `bound=4096` with `senses=` and `code=` |
| A voice that says the sound standing for what it sees | a voice | `voice="ear"`, `brain.say(...)` |
| A said sound that becomes the answer once paid | a reward | `brain.live(..., reward=[1.0])` |
| A heard name that draws the focus to its thing | the named focus | `bound=`, `voice=`, `surprise=True` and a map with values |
| Learning while young, surprised or in want, calm answers otherwise | [arousal](#learning) | `arousal=True`, then `brain.live(...)` |
| A teacher who shows the answer | lessons | `brain.lesson(observations, labels)` |
| The same brain tomorrow | [save and load](#saving-and-loading) | `brain.save(path)`, `Brain.load(path)` |
| An Apple or NVIDIA GPU, many brains at once | [devices](#devices) | `backend="mlx"`, `cd.Lives(...)` |

Four whole brains, from the plainest to one with eyes, an ear, a voice and a map, are
[below](#four-assemblies).

## The parts

### Senses

A sense is declared once, by name, with its width, or with a width for each of its fields.
Every call that takes observations then takes a dict of sense to rows; a sense left out is
silent. `cadence.senses` holds two senses to start from: `Things` gives each field of a seen
thing as tuned cells (eight a reading), and `Ear` gives a heard utterance, onset to the quiet
after it, as one code from its log-mel frames.

```python
import numpy as np
import cadence as cd
from cadence import Brain
from cadence.senses import Ear, Things

eyes = Things({"hue": (0.0, 1.0), "shape": [(0.0, 1.0)] * 3})   # a hue and three form readings
ear = Ear(frames=8, bands=10)                                     # a word as one code
creature = Brain.compose(senses={"eyes": eyes.widths, "ear": ear.width}, actions=2, seed=0)
apple = eyes.rows({"hue": [0.05], "shape": [[0.5, 0.5, 0.5]]})
word = ear.code(np.random.default_rng(0).normal(size=(20, 10)))   # 20 frames of 10 bands
action = creature.act({"eyes": apple, "ear": word}, greedy=True)
action = creature.act({"eyes": apple}, greedy=True)               # the ear is silent
assert creature.inputs == 8 + 24 + 80                             # the senses side by side
```

- **Equal senses.** With `code=units` every field of every sense enters the brain's code at unit
  length, so a word of one input counts as much as a glimpse of fifty. Use it whenever senses
  differ in width or loudness, and always for records and a voice
  ([equal senses](senses.md#equal-senses-in-the-fixed-code)).
- **The senses beside the code.** With `code=units` the cortex reads the code alone, which holds
  what is on together and holds a bearing or a distance less directly. A body that acts on
  where things are from it reads its senses beside the code (`cortex_reads="both"`, below).
- **Other senses.** A few continuous readings go through `cd.PopulationCode`; an image is
  `inputs=(height, width)`, read best through `learned_fields`
  ([senses](senses.md)). `Things` and `Ear` learn nothing and are not saved with the
  brain: make them again with the same arguments.

#### A body that acts on what it senses

An animal's senses: eyes in eight sectors, ears, a call it heard, the thing it attends, the
nearest food (what it is, its bearing, its distance) and its own body. The code of 128 units is
where its records key; with `cortex_reads="both"` its cortex reads the senses beside the code,
so a lesson to turn toward the food reaches the bearing itself
([what the cortex reads](senses.md#what-the-cortex-reads)).

```python
animal = {"eyes": 32, "ears": 8, "heard": 16, "thing": 20,
          "food": {"what": 8, "bearing": 8, "distance": 4}, "body": 8}
forager = Brain.compose(senses=animal, actions=5, code=128, cortex_reads="both",
                        modules=(32, 32), bound=16, voice="heard", arousal=True, seed=0)
food_left = {"food": {"what": np.eye(8)[[0]], "bearing": np.eye(8)[[2]],
                      "distance": np.eye(4)[[1]]}}
forager.lesson(food_left, np.array([2]))                          # a teacher shows: turn left
action = forager.live(food_left)
assert len(forager.connectome.populations["observation"]) == forager.inputs == 104 + 16
```

The `observation` region holds every input, the voice the animal hears itself say among them.

### Actions and a voice

`actions=n` gives one choice a moment among `n`. `slots=(3, 2)` splits the actions into parts
chosen together (a motor and a grip), one label per slot in a lesson
([motor competition](tune.md#operating-point-and-motor-competition)). A voice is a body part of
its own: `voice="ear"` lets the brain say sounds in the ear's format, read from its records of
bound moments, and adds a sense `voice` through which it hears what it says
([symbols](#symbols)).

### The map of things

The map holds where every thing is, each place bound to what the thing is, and keeps a thing at
its last known place while nothing sees it. `trained=["place"]` gives a brain the map the library
ships; `trained=[Map(...)]` gives it a map of your own. Either way the map becomes `brain.space`,
the eyes report to it with `brain.space.moment(seen=..., lit=...)`, and the brain reads it by
itself at every moment as one more sense, `space`. The name `trained=` is for parts that arrive
whole; `modules=` sets the width of the brain's own regions and is unrelated.

```python
from cadence.space.map import Map

APPLE, BALL = 0, 1
room = Map((-1.0, 1.0, -1.0, 1.0), (16, 16), units=8, values=eyes.widths)   # hue and shape
room.moment(seen=[(APPLE, [-0.5, 0.0], {"hue": apple["hue"][0], "shape": apple["shape"][0]})])
for _ in range(10):
    room.moment(lit=lambda points: points[:, 0] > 0.0)           # the eyes look right
held = room.thing_at([-0.5, 0.0])                                 # the apple, out of view
assert held["kind"] == APPLE and held["values"]["hue"].argmax() == 0
```

A report is a kind (a number), a place and, for a map with values, one row a field named as the
eyes' fields: `Things.rows` gives rows a stream, so a report takes row `[0]` of each.

| You want | Use | Guide |
| --- | --- | --- |
| Where a thing is, seen or not; what is at a place | `where_is(kind)`, `what_is_at(place)` | [space](space.md#questions-answered-by-settling) |
| Each thing's colour and form carried with it | `Map(plane, values={...})`, `thing_at` | [values carried with a thing](space.md#values-carried-with-a-thing) |
| A hidden thing that moves, followed | `Map(plane, physics=True)`, `forecast` | [learned physics](space.md#learned-physics-on-the-map) |
| A thing's move as surprise, drawing the focus | `Map(plane, surprise=True)` with `surprise=True` | [surprise on the map](space.md#surprise-on-the-map-opt-in) |
| A trained brain's regions in a new brain | `save_module`, `trained=[module]` | [modules](modules.md#modules-parts-that-arrive-as-files) |

### Memory

A new brain holds four memories without being asked. Two more are opt-in.

| Memory | Holds | In a brain |
| --- | --- | --- |
| Working trace | what the brain was just thinking | every brain |
| Heard history | what it heard a moment ago | every new brain; `history_decays=()` builds none, and `recall=False` leaves it out with the recall memory |
| Recall memory | 128 recent moments a stream, back by their content | every new brain; `recall=False` builds none |
| Associative store | what paid in each situation | every brain; `episodic=False` builds none |
| Long-term records | facts told once and asked long after, read by several learned heads | `longterm=records` with a held message ([long-term records](memory.md#long-term-records)) |
| Records of bound moments | what the senses held together, read back by any part | `bound=records` ([symbols](#symbols)) |

A memory's strength follows one law wherever it is set: a record told again or written at a
moment of surprise grows deep, one deep enough is kept for good, and one never used fades
([a record's strength](memory.md#a-records-strength-opt-in)). It is on for the records of bound
moments and opt-in for the others (`recall_strength=True`, `longterm_strength=True`). The
[memory guide](memory.md#which-memory-should-i-use) says which memory serves which task.

### Attention

`surprise=True` gives a brain one surprise signal, read from its own settle: how far the moment's
input left the state the brain carried into it, against what each neuron is used to.
`brain.surprise.moment` holds one number a stream, zero when the moment brought nothing new. The
same signal deepens the records written at that moment and draws the focus: `brain.focus` is the
one place attended, among the places of the senses (`brain.places`) and the cells of a map with
`surprise=True`. A name heard by a brain with a voice and a map with values draws the focus to
the thing it names ([eyes, an ear, a voice and a map](#eyes-an-ear-a-voice-and-a-map)). The library reports the focus; turning the body
toward it is the program's.

```python
watcher = Brain.compose(senses={"eyes": eyes.widths}, actions=2, surprise=True, seed=0)
for _ in range(8):
    watcher.act({"eyes": apple}, greedy=True)                     # the same apple, again
assert watcher.surprise.moment[0] == 0.0                         # nothing new
ball = eyes.rows({"hue": [0.6], "shape": [[0.9, 0.9, 0.1]]})
watcher.act({"eyes": ball}, greedy=True)
assert watcher.surprise.moment[0] > 0.0                          # a ball where the apple was
```

The focus's rules (how long it holds, how long a place it left stays inhibited) are genes of
`surprise=` ([surprise, and the focus it draws](life.md#surprise-and-the-focus-it-draws-opt-in)).

### Learning

A brain learns while it lives. A lesson shows the right answer; a reward scores the brain's own
last action. With `arousal=True`, `live` learns while the brain is young (its first 100 moments),
surprised by an outcome (arousal's own measure, apart from `surprise=True`) or wanting, and
answers calmly from one settle otherwise.

```python
pupil = Brain.compose(senses={"eyes": eyes.widths}, actions=2, arousal=True, seed=0)
pupil.lesson({"eyes": apple}, np.array([1]))                      # a teacher shows the answer
action = pupil.live({"eyes": apple})                              # it acts
action = pupil.live({"eyes": ball}, reward=[1.0])                 # the world pays that act
assert pupil.learner.updates > 0
```

| You have | Use | Guide |
| --- | --- | --- |
| Answers for recorded observations | `lesson`, `step(teacher=...)` | [train a brain](teach.md) |
| A world that pays | `live(..., reward=...)`, or `step` for many streams | [learning from reward](reward.md) |
| A lesson that names some slots and leaves one open | a label of `-1` | [a slot the lesson leaves open](teach.md#a-slot-the-lesson-leaves-open) |
| A reward for something said | the reward pays the record of the moment it acted at | [symbols](#symbols) |

### Symbols

`bound=records` (with `senses=` and `code=`) gives a brain records of what its senses held
together. Every moment it lives with something in a sense is written; a moment told again
deepens its record. Before each settle, a sense left silent is filled with what the nearest
records hold alike, and the brain settles on the moment so completed: "cup" heard alone brings
the cup's form back into the eyes. `brain.complete(observations)` shows what comes back and
changes nothing.

With `voice="ear"`, `brain.say(observations)` gives the sound that stands for the moment, in the
ear's format. Pass it back under `"voice"` at the moment it is said, so the moment's record holds
it. A reward then pays that record: a said sound becomes the answer only once paid, and a reward
of zero or less refuses it. `brain.bound.said_from` names where each sound came from: `"said"` (a
paid answer), `"heard"` (what was heard with things like this), `"babble"` (a sound it knows,
tried) or `None` (no sound held yet).
The [memory guide](memory.md#records-of-bound-moments-opt-in) gives the rules of the read, the
answer and the pay.

### Saving and loading

`brain.save(path)` writes the whole continuation into one file: weights, every memory and record,
the map, the names of the senses and an action that awaits its outcome. `Brain.load(path)` goes
on from there.

```python
again = Brain.load(creature.save("creature.npz"))
assert again.senses == creature.senses
assert again.act({"eyes": apple}, greedy=True) == creature.act({"eyes": apple}, greedy=True)
```

A file names the features it needs, such as `bound` or `surprise`, and a release that lacks one
refuses the file by that name. A brain without them saves in the format the release before reads
([checkpoint format](api.md#checkpoint-format)).

### Devices

A new brain picks where it runs by its size at its first moment: small brains on the CPU, wide
ones on an Apple or NVIDIA GPU when one is installed (`[apple]`, `[accel]`). `backend="cpu"`
repeats to the bit on every run ([which backend](backends.md#which-backend-should-i-use)).
`surprise=True` runs on the CPU.

- **Threads.** Numba's threads give the same bits at any count. Many brains in a process pool
  run fastest with one thread each (`NUMBA_NUM_THREADS=1`). A reloaded life takes the same
  actions on any machine; for one that repeats to the last bit, use the NumPy path
  (`CADENCE_FUSED=0`) with `OMP_NUM_THREADS=1` set before NumPy is imported
  ([threads](backends.md#a-reloaded-brain-and-blas-threads)).
- **Many brains at once.** `cd.Lives(inputs, actions, lives)` steps brains of their own together.
  It takes one row of inputs a life, so senses by name, a map and records belong to lone brains
  ([many lives at once](backends.md#many-lives-at-once)).
- **A browser page.** The library needs NumPy alone, so it runs under Pyodide, on the NumPy path;
  the map's timings there are in [the space guide](space.md#readings).

## Four assemblies

Each brain below is a few lines, with nothing set but the parts it holds.

### A plain brain

Senses, a motor, the default memories, learning while it lives.

```python
import numpy as np
from cadence import Brain

brain = Brain.compose(inputs=4, actions=2, arousal=True)
action = brain.live(np.array([[1.0, 0.0, 0.0, 0.0]]))
reward = (action == 0).astype(float)                              # the world pays action 0
action = brain.live(np.array([[0.0, 1.0, 0.0, 0.0]]), reward=reward)
```

### A brain with a map that keeps hidden things

Touch is the brain's own sense; the eyes report to the map, which keeps the apple while the eyes
look away. The brain reads the map at every moment.

```python
import numpy as np
from cadence import Brain

APPLE = 0

def ahead(points):                                               # the eyes look to the right
    return points[:, 0] > 0.0

def behind(points):                                              # and then to the left
    return points[:, 0] < 0.0

kitchen = Brain.compose(senses={"touch": 8}, actions=2, arousal=True, trained=["place"])
touch = np.eye(8)[[2]]
kitchen.space.moment(seen=[(APPLE, [0.3, 0.6])], lit=ahead)      # the eyes report an apple
action = kitchen.live({"touch": touch})                          # the brain reads its map
for _ in range(20):
    kitchen.space.moment(lit=behind)                             # the eyes look away
action = kitchen.live({"touch": touch}, reward=[1.0])
assert np.allclose(kitchen.space.where_is(APPLE), [[0.3, 0.6]], atol=0.001)   # held unseen
```

The map's read is 848 values of small size beside the touch. Add `code=256` when the brain's own
task depends on the map: its memories then key on 256 units, and the map counts as much as the
touch ([a brain that knows where things are](modules.md#a-brain-that-knows-where-things-are)).

### A brain with an ear and a voice

Its sounds stand for its things. Shown a cup while it hears "cup", it keeps the record; "cup"
heard alone brings the cup back, the cup seen alone makes it say "cup", and a paid answer is the
one it gives from then on.

```python
import numpy as np
from cadence import Brain
from cadence.senses import Ear, Things

eyes = Things({"hue": (0.0, 1.0), "shape": [(0.0, 1.0)] * 3})
ear = Ear(frames=8, bands=10)
talker = Brain.compose(senses={"eyes": eyes.widths, "ear": ear.width}, actions=2,
                       code=512, bound=256, voice="ear", arousal=True)
rng = np.random.default_rng(0)
cup = eyes.rows({"hue": [0.85], "shape": [[0.6, 0.3, 0.2]]})
cup_word = ear.code(rng.normal(size=(20, 10)))                   # "cup", as log-mel frames
for _ in range(3):
    talker.act({"eyes": cup, "ear": cup_word}, greedy=True)       # a cup, and "cup" heard
assert talker.complete({"ear": cup_word})["eyes"]["shape"].any()  # "cup" alone: the cup's form
said = talker.say({"eyes": cup})                                  # a cup alone: what it says
assert float(said[0] @ cup_word[0]) > 0.9                         # the word it heard with it

question = {"eyes": cup, "ear": ear.code(rng.normal(size=(20, 10)))}   # "what is this?"
said = talker.say(question)
talker.live({**question, "voice": said})                          # it answers, and hears itself
talker.live({"eyes": cup}, reward=[1.0])                          # paid: that is the answer
assert np.allclose(talker.say(question), said) and talker.bound.said_from == ["said"]
```

### Eyes, an ear, a voice and a map

Records, surprise and a map with each thing's hue and shape. A trainer names two things; then a
screen hides the ball, and "ball" heard alone turns the brain's attention to the place behind
the screen where the ball is.

```python
import numpy as np
from cadence import Brain
from cadence.senses import Ear, Things
from cadence.space.map import Map

eyes = Things({"hue": (0.0, 1.0), "shape": [(0.0, 1.0)] * 3})
ear = Ear(frames=8, bands=10)
room = Map((-1.0, 1.0, -1.0, 1.0), (16, 16), units=8, values=eyes.widths, surprise=True)
brain = Brain.compose(senses={"eyes": eyes.widths, "ear": ear.width}, actions=2, code=512,
                      bound=256, voice="ear", surprise=True, trained=[room], arousal=True)

APPLE, BALL = 0, 1
rng = np.random.default_rng(1)
apple = eyes.rows({"hue": [0.05], "shape": [[0.5, 0.5, 0.5]]})
ball = eyes.rows({"hue": [0.6], "shape": [[0.9, 0.9, 0.1]]})
apple_word, ball_word = ear.code(rng.normal(size=(20, 10))), ear.code(rng.normal(size=(20, 10)))
for _ in range(3):                                                # the trainer names each thing
    brain.act({"eyes": apple, "ear": apple_word}, greedy=True)
    brain.act({"eyes": ball, "ear": ball_word}, greedy=True)


def report(kind, place, thing):                                   # what the eyes tell the map
    return (kind, place, {field: rows[0] for field, rows in thing.items()})


def left_only(points):                                            # a screen hides the right
    return points[:, 0] < 0.0


brain.reset()                                                     # a new scene; records stay
room.moment(seen=[report(APPLE, [-0.5, 0.0], apple), report(BALL, [0.5, 0.0], ball)])
brain.act({}, greedy=True)                                        # it looks at the room
room.moment(seen=[report(APPLE, [-0.5, 0.0], apple)], lit=left_only)   # the ball is hidden
brain.act({"ear": ball_word}, greedy=True)                        # "ball!"
assert brain.focus.sense == "named"                               # the name drew the focus
assert np.allclose(brain.focus.place, [0.5, 0.0])                 # to the ball, behind the screen

said = brain.say({"eyes": apple})                                 # an apple in view
assert float(said[0] @ apple_word[0]) > 0.9                       # "apple"
brain.live({"eyes": apple, "voice": said})                        # it says it
brain.live({"eyes": apple}, reward=[1.0])                         # paid
assert brain.bound.paid.max() > 0
```

The [symbols chamber](../benchmarks/symbols/README.md) measures these parts on twelve fresh
founders in a larger brain (`code=2048`, `bound=4096`, `modules=(256,)`): with the default 64
association neurons an apple and a ball, alike in form and both red, settle alike.

## What each part costs

Measured on an Apple M4 shared with other work, one thread, the CPU backend with Numba installed
(`[fast]`; with NumPy alone a moment takes about three times as long), the median of 30 to 60
moments. A `live` moment here learns (the brain is young); a calm one costs about an `act`.

| Brain | An `act` | A `live` that learns | Parameters | Saved file |
| --- | --- | --- | --- | --- |
| A plain brain (4 inputs, 2 actions) | 1.1 ms | 9 ms | 14,233 | 0.14 MB |
| With the map (`trained=["place"]`) | 3.4 ms, and 4.8 ms a map moment | 32 ms | 1,130,353 | 3.6 MB |
| With an ear and a voice (`code=512`, `bound=256`) | | 19 ms, 28 ms with 251 records | 504,961 | 3.4 MB |
| Eyes, an ear, a voice and a map, 251 records | | 37 ms, and 5.3 ms a map moment | 504,961 | 3.3 MB |
| The symbols chamber's brain (`code=2048`, `bound=4096`, `modules=(256,)`) | | 217 ms | 7,180,417 | 47 MB |

- **Width costs most.** The recall memory's maps grow with the square of what it keys on: the
  map's 848 values beside the touch take the brain from 14 thousand to 1.1 million parameters.
  `code=units` bounds that width ([recall cost](memory.md#recall-cost)).
- **Records cost a read a moment.** `complete` read 2.8 ms and `say` 17 ms among 251 records.
- **The first moment compiles.** With Numba installed (`[fast]`), the first moment of a new
  source version compiles the kernels, about 10 seconds on an idle M4, and loads them from the
  cache after that.

## What it does not do

- **One thing at a time is bound.** The settled state holds the thing in focus with all it is
  (kind, colour, place, the word heard with it); the other things in a scene are held by the map,
  and a question is answered about the thing in focus. Readings on
  [#342](https://github.com/muellerberndt/cadence/issues/342).
- **The settle alone does not complete a thing from one sense.** Without records, a word heard
  alone brings up its thing at chance, and one sense of a thing reads even its own facet poorly
  ([#342](https://github.com/muellerberndt/cadence/issues/342)). The records of bound moments do
  it: a name heard alone brings its thing back at 1.00 in the symbols chamber.
- **Names from looking alone are partial.** A thing seen alone is named from moments of naming
  without reward at .80 to .95 in the symbols chamber; those moments do not tell the records
  which fields a name follows. A reward does.
- **The voice says sounds it has heard.** `say` picks among the sounds its records hold; it forms
  no new sound.
- **A name does not draw the focus back at once.** A place the focus has just left is inhibited
  for a while (`surprise={"inhibit": ..., "release": ...}`), and a name does not overcome it.
- **What refuses what.** `grow`, `graft` and `Lives` refuse a brain with records; `Lives` takes
  no senses by name and no map; `surprise=True` runs on the CPU. The whole list is
  [what does not go together](modules.md#what-does-not-go-together).
- **Sentences and records at any size are work in progress.** A sentence held in the state
  ([#308](https://github.com/muellerberndt/cadence/issues/308)) and long-term records past a
  machine's memory ([#310](https://github.com/muellerberndt/cadence/issues/310)) are in no
  release ([the roadmap](https://github.com/muellerberndt/cadence/issues/109)).
