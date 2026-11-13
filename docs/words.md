# Words for things

A brain with a map of its room learns words by being shown things and told their names, and then
answers typed questions about the room in those words: "where is the ball?", "what colour is the
ball?", "what is red?", "what is behind you?". It answers from what it holds, including things it
cannot see, and says "I do not know" where it holds no answer.

The module is `cadence.words`, opt-in: a program that does not import it gets a library without
it. It has two parts.

| Part | What it is | Section |
| --- | --- | --- |
| `Lexicon` | Words bound to the concepts a brain holds, learned from naming | [the lexicon](#the-lexicon) |
| `Words` | A brain that answers questions about its room: a map with colours, a body and a lexicon | [a brain that answers](#a-brain-that-answers-about-its-room) |

Both rest on the map of things ([space](space.md)), whose sheet also carries what each thing is
like ([values carried with a thing](space.md#values-carried-with-a-thing)).

## A first brain

```python
import math
import numpy as np
from cadence.space.map import Map
from cadence.words import Words

BALL, CUP, BOX = 0, 1, 2
RED, BLUE = 0, 1
room = Map((-2.0, 2.0, -2.0, 2.0), (24, 24), units=3, values={"colour": 2})
brain = Words(room)                                    # the parrot perches at (0, 0)
kinds, colours = ["ball", "cup", "box"], ["red", "blue"]
places = ["front", "right", "behind", "left"]          # the four places around the body
rng = np.random.default_rng(0)


def new_room():
    """Three things around the parrot, each in a drawn colour, seen once."""
    turn = rng.uniform(0.0, 2 * math.pi)
    things = [(k, 1.2 * np.array([math.cos(turn + 2.1 * k), math.sin(turn + 2.1 * k)]),
               int(rng.integers(2))) for k in range(3)]
    room.reset()
    room.moment(seen=[(k, p, {"colour": c}) for k, p, c in things])
    return things


for _ in range(40):                                    # naming: show a thing, say a word
    things = new_room()
    for _ in range(4):
        k, p, c = things[int(rng.integers(3))]
        brain.body.heading = float(rng.uniform(-math.pi, math.pi))
        u = rng.random()
        word = kinds[k] if u < 0.5 else colours[c] if u < 0.75 else places[brain.body.place(p)]
        brain.name(word, p)                            # the brain attends to the thing at p
brain.body.heading = 0.0
for _ in range(4):                                     # practice: questions with their answers
    things = new_room()
    k, p, c = things[0]
    brain.practice(f"where is the {kinds[k]}", places[brain.body.place(p)])
    brain.practice(f"what colour is the {kinds[k]}", colours[c])
    brain.practice("where is the zebra", "i do not know")

things = new_room()                                    # a room it has never seen
k, p, c = things[1]
assert brain.ask(f"where is the {kinds[k]}") == [places[brain.body.place(p)]]
assert brain.ask(f"what colour is the {kinds[k]}") == [colours[c]]
assert brain.ask("where is the kettle") == ["i do not know"]   # a word never named
```

The brain was never told which word names a kind, which a colour and which a place: every
naming showed a whole thing, and the word took what its namings shared.

## What the brain does and what the program does

| | The program | The brain |
| --- | --- | --- |
| A naming | Passes the word and the place attended to | The map settles on what is there (its kind and colour), the body gives the place around it, and the lexicon learns the word with those concepts by the library's free and nudged phases |
| A question | Splits the text into words | The lexicon settles on the words: the concept asked about (the most active kind, colour or place) and each read's activity (what is asked for) |
| Finding the things | Sets the map's cue from the concept the lexicon settled on | The map settles on the things: a kind's places, a colour's places, what is at a place around the body |
| The answer | Joins the letters the motor said | For each thing found, its concepts, every group times its read's settled activity, drive the lexicon's saying settle; the motor chooses the most active word, hears it, and says it where hearing it gives back the concept said |
| Knowing nothing | Nothing | A word never heard drives no concept; a concept the map does not hold finds nothing; both give the read that finds nothing, whose word the teacher said once |

No table from word to concept is held anywhere, and none from a question word to a read: a
word's meaning is in the synapses between its neuron and the concept neurons, a question word's
read is in the synapses its practice taught it, the reads act as gains on what was found, and an
answer is said by the motor. With the lexicon's synapses at zero the same brain answers nothing.

## The lexicon

`Lexicon(concepts)` takes the concepts as named groups and their sizes, `{"kind": 12,
"colour": 6, "place": 4}`:

- **A word enters when it is first heard.** It takes the next free word neuron, and the motor's
  letters for it are written from the letters it was heard by. A form never heard has no neuron.
- **Naming.** `name(word, {("kind", 3): 0.4, ("colour", 1): 0.3})`: the word's neuron is driven,
  the concept neurons settle, and the nudge pulls them toward the concepts present at that moment,
  each at its share. A word's n-th lesson closes 1/n of its error (no less than `rate`), so its
  synapses hold the mean of its namings: a concept present at every naming is held at its share,
  one present at a third of them at a third of it.
- **Hearing.** `hear(words)`: the words' neurons are driven and the concept neurons settle.
- **Saying.** `say(("colour", 1))`: the concept's neuron is driven, the word neurons settle and
  the motor chooses the most active word and says its letters. The synapses of a word and a
  concept are one reciprocal pair, so hearing and saying read the same synapses: no lesson
  teaches the way back. Before it says a word the brain hears it: a word whose hearing settles
  on another concept of the groups said gives way to the next most active (a new kind's word,
  named three times on one red thing, is bound to red too, and the word "red" is said for red).
- **A threshold on the word neurons** (`threshold`, 0.15) keeps a settle from spreading from word
  to word: a word is active where it is heard or where a driven concept names it.

```python
from cadence.words import Lexicon

lexicon = Lexicon({"kind": 4, "colour": 3})
for c in (0, 1, 2):                                    # a ball in three colours
    lexicon.name("ball", {("kind", 0): 0.5, ("colour", c): 0.5})
for k in (1, 2, 3):                                    # three kinds of thing, each red
    lexicon.name("red", {("kind", k): 0.5, ("colour", 0): 0.5})
assert int(lexicon.hear("ball").argmax()) == lexicon.index("kind", 0)
assert int(lexicon.hear("red").argmax()) == lexicon.index("colour", 0)
assert lexicon.say(("kind", 0)) == "ball" and lexicon.say(("colour", 0)) == "red"
assert (lexicon.hear("kettle") < 0.05).all()           # never heard: names nothing
```

## A brain that answers about its room

`Words(map)` takes a map (with `values=` for what things are like) and builds its senses
(`Senses`: the map's kinds, each group of the map's values and the four places around the body:
in front, to the right, behind, to the left), and a lexicon over those groups, one read a group
and the read that finds nothing. `brain.body` is where it stands and which way it faces.

- `look(seen=, lit=)`: one moment of the senses, passed to the map.
- `name(word, at)`: a naming of the thing at `at`, each group at its share of `shares` (the kind
  .4, the other groups an even split of the rest: .3 and .3 with one group of values).
- `ask(question)`: the words said, one a thing found.
- `practice(question, answer)`: a question with the teacher's answer. The brain hears the answer,
  reads in its own lexicon which group it names (a place, a colour, a kind), and the question's
  words are taught that read. An answer that names no concept, given where the brain's read finds
  nothing, is named with the read that finds nothing: that is how "i do not know" is learned.

**Both directions.** A question names a concept and asks for one. "What colour is the ball?" goes
from a thing to its colour: the map settles on the ball's places (`where_is`) and reads the colour
there (`values_at`). "What is red?" goes from the colour to the things: the colour's unit opens its
pool and the map settles on the places of red things (`where_is_value`); each one's kind is read
there (`what_is_at`). "What is behind you?" goes from a place around the body to the things: the
map settles on what is on the cells behind the body (`what_is_at(cells=)`), one thing at a time.
Every direction reads the same lexicon and the same map.

## Pistachios: what a question asks for, from reward

A question word is learned the way a noun is, from questions answered for the brain, and the
answer need not be told: a right answer can earn a pistachio. `ask(question, explore=True)` tries
a read (what to say of the things found: their kind, their colour or their place), drawn from the
reads' settled activity at `temperature`; `reward(1.0)` gives the pistachio. The question's words
of asking, those that name no concept of the senses, are taught the read the answer took, at the
reward's surprise: the reward less the chance the brain gave that read. A word that names a kind,
a colour or a place keeps no read, so a read learned with one colour word holds for all of them,
and no lesson of reward touches what a word names. A wrong answer earns nothing and teaches
nothing.

```python
def raised_on_pistachios(rounds=30):
    """The first brain's namings, then questions answered only with pistachios."""
    parrot = Words(Map((-2.0, 2.0, -2.0, 2.0), (24, 24), units=3, values={"colour": 2}), seed=1)
    room_ = parrot.map

    def room_of():
        turn = rng.uniform(0.0, 2 * math.pi)
        things = [(k, 1.2 * np.array([math.cos(turn + 2.1 * k), math.sin(turn + 2.1 * k)]),
                   int(rng.integers(2))) for k in range(3)]
        room_.reset()
        parrot.look(seen=[(k, p, {"colour": c}) for k, p, c in things])
        return things

    for _ in range(40):
        things = room_of()
        for k, p, c in [things[int(rng.integers(3))] for _ in range(4)]:
            parrot.body.heading = float(rng.uniform(-math.pi, math.pi))
            u = rng.random()
            parrot.name(kinds[k] if u < 0.5 else colours[c] if u < 0.75 else
                        places[parrot.body.place(p)], p)
    parrot.body.heading = 0.0
    for _ in range(rounds):
        k, p, c = room_of()[0]
        for question, right in ((f"where is the {kinds[k]}", places[parrot.body.place(p)]),
                                (f"what colour is the {kinds[k]}", colours[c])):
            if parrot.ask(question, explore=True) == [right]:
                parrot.reward(1.0)                     # a pistachio for a right answer
    return parrot, room_of()


parrot, things = raised_on_pistachios()
k, p, c = things[2]
assert parrot.ask(f"what colour is the {kinds[k]}") == [colours[c]]
```

In the [chamber](../benchmarks/words/PROTOCOL_2.md) a parrot that practises on pistachios alone
is read beside one told the answers and one whose pistachios are withheld.

## Parts that plug in

The question path is open to parts that add what the brain holds and how a question finds it
(records of the past, a forecast of what comes, relations between things, counting). A part is
an object with `groups` (its concept groups, added to the lexicon) and any of:

| Member | Called |
| --- | --- |
| `answers` | Those of its groups a question may ask for: each gets a read |
| `sense(brain, seen, lit)` | At each moment `look` passes to the map, before the map takes it |
| `find(brain, heard, things)` | After the senses found the things a question names: gives the things found, each `(place, concepts by group)` |
| `practice(brain, words, answer, heard)` | After a question answered: the part's own lessons |

`Words(map, parts=[...])` takes them in order, and `Words(map, senses=...)` takes learned senses
in place of the born ones: an object with `groups`, `describe(place)` and `find(group, pattern)`.

## Concepts the parrot learns

The given senses hand the brain its categories: a unit a kind, a unit a colour, the four quarters
around the body. `cadence.words.concepts.Concepts` replaces them with concepts the brain learns. It
is born with senses and no category:

- the **hue** of a thing and the **direction** of a place from where the body faces, each over
  inputs tuned around the circle (`Circle`), as hue-tuned and head-direction cells tile theirs;
- a thing's **shape and size**, three features each over tuned inputs, an input a combination
  (`Features`).

Each sense feeds a **region** of concept neurons that start silent; the region settles and its most
active neuron is the concept. A kind concept enters where a thing's shape is explained by none (a
thing of a new shape is a new kind before any word names it). A colour or a place concept enters by
naming. **Naming moves the boundaries**: a word's image of each sense (the mean reading of its
namings) says which sense the word names, "red" the hue, "ball" the kinds, "behind" the direction.
A naming then pulls the word's own concept up on the reading and every other concept of that
region to rest, by the library's free and nudged phases. So the concept of red ends where this
teacher stops saying "red", and a hue never shown is named by the nearest concept. A new word heard
on a thing whose kind, colour and place each carry another word's name takes a new kind
concept: a new word goes to what has no name.

```python
import numpy as np
from cadence.space.map import Map
from cadence.words.concepts import Concepts, LearnedWords

room = Map((-2.0, 2.0, -2.0, 2.0), (16, 16), units=6, values={"colour": 4})
concepts = Concepts(room, [(0.0, 1.0)] * 3, places=4)   # three shape features, each in 0 to 1
parrot = LearnedWords(room, concepts)
rng = np.random.default_rng(0)
shapes = {"dako": np.array([0.2, 0.3, 0.4]), "mipu": np.array([0.7, 0.6, 0.5])}
edges, colours = [0.0, 1.0, 3.0, 4.6], ["ruka", "beto", "gani", "sefo"]  # this teacher's hues
for _ in range(320):                                     # namings of a thing in front
    kind = str(rng.choice(list(shapes)))
    hue, at = float(rng.uniform(0.0, 2 * np.pi)), np.array([1.2, rng.uniform(-0.6, 0.6)])
    room.reset()
    parrot.look([(shapes[kind] + rng.normal(0.0, 0.03, 3), at, hue)])  # what the eyes report
    colour = colours[int(np.searchsorted(edges, hue, side="right")) - 1]
    parrot.name(kind if rng.random() < 0.4 else colour, at)
concepts.learning = False
assert concepts.regions["colour"].used == 4             # one colour concept a colour word
said = [parrot.lexicon.say(("colour", concepts.percept(shapes["dako"], h)[1]))
        for h in (0.6, 0.9, 1.1, 1.4)]
assert said == ["ruka", "ruka", "beto", "beto"]         # its boundary is this teacher's, at 1.0
```

`LearnedWords` is `Words` with these senses: `look` takes the eyes' raw reports `(shape, place,
hue)`, and `name` takes the concepts' lesson before the lexicon's. The lexicon then teaches a word
the concept of its own sense alone, and leaves at rest a concept another word of that sense names
more. [`benchmarks/words/concepts.py`](../benchmarks/words/concepts.py) gives each founder a
teacher whose boundaries move, holds hues and shapes out of training, and reads where the parrot's
boundaries lie ([its declared protocol](../benchmarks/words/CONCEPTS_PROTOCOL.md)).

| Setting | Default | What it sets |
| --- | --- | --- |
| `Concepts(places=)` | 10 | Place concept neurons (the kinds and colours are the map's units) |
| `hues`, `directions`, `per` | 48, 48, 8 | Tuned inputs of the hue, the direction and each shape feature |
| `vigilance` | 0.5 | The share of its level under which a reading is unexplained (kinds) |
| `rate` | 0.3 | The share of its error a naming's lesson closes in a region |
| `margin`, `sure`, `least` | 1.5, 0.1, 5 | How far a word's image must stand apart to name a sense, and from how many readings |
| `naming` | True | False: the regions learn from the senses alone (a control) |
| `exclusive` | True | A naming leaves at rest a concept another word of its sense names more |

## The chamber and its readings

[`benchmarks/words`](../benchmarks/words/README.md) raises a brain over rooms of nonce words drawn
for each founder and asks it every question form, beside an LSTM and a transformer given the same
namings and questions ([the declared protocol](../benchmarks/words/PROTOCOL.md)).

## Settings and cost

| Setting | Default | What it sets |
| --- | --- | --- |
| `Lexicon(capacity=)` | 256 | Word neurons |
| `rate` | 0.05 | The least share of its error a word's lesson closes |
| `threshold` | 0.15 | A word neuron's threshold |
| `steps` | 12 | Sweeps of a settle |
| `Words(shares=)` | the kind 0.4, the rest evenly | A naming's shares of the senses' groups |
| `floor` | 0.05 | The activity a concept needs to count as named |
| `sharpness` | 0.02 | How the reads compete for their gains when the brain answers |
| `temperature` | 0.1 | How a read is drawn where the brain tries one (`ask(explore=True)`) |

A settle of the lexicon reads every synapse once a sweep: 12 sweeps over 13,312 synapses at the
defaults, about 160,000 multiply-accumulates. A question takes one settle to hear, one or more
settles of the map and one settle a word said.
