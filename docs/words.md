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
brain = Words(room)                                    # the robot stands at (0, 0)
kinds, colours = ["ball", "cup", "box"], ["red", "blue"]
places = ["front", "right", "behind", "left"]          # the four places around the body
rng = np.random.default_rng(0)


def new_room():
    """Three things around the robot, each in a drawn colour, seen once."""
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
| A question | Splits the text into words | The lexicon settles on the words: what is asked about (a kind, a colour or a place) and what is asked (its kind, its colour or its place) |
| Finding the things | Sets the map's cue from the concept the lexicon settled on | The map settles on the things: a kind's places, a colour's places, what is at a place around the body |
| The answer | Joins the letters the motor said | For each thing found, the lexicon settles from the read's concept to a word, and the motor says the most active word's letters |
| Knowing nothing | Nothing | A word never heard drives no concept; a concept the map does not hold finds nothing; both give the read that finds nothing, whose word the teacher said once |

No table from word to concept is held anywhere: a word's meaning is in the synapses between its
neuron and the concept neurons, and an answer is said by the motor. With those synapses at zero
the same brain answers nothing.

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
  teaches the way back.
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

`Words(map)` takes a map made with `values=` (the colour by default, `group=` names another) and
builds a lexicon over the map's kinds, the values, the four places around the body (in front, to
the right, behind, to the left), the three reads and the read that finds nothing. `brain.body` is
where it stands and which way it faces.

- `name(word, at)`: a naming of the thing at `at`, each concept at its share of `shares` (the
  kind .4, the value .3, the place around the body .3).
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
| `Words(shares=)` | (0.4, 0.3, 0.3) | A naming's shares of the kind, the value and the place around the body |
| `floor` | 0.05 | The activity a concept needs to count as named |

A settle of the lexicon reads every synapse once a sweep: 12 sweeps over 13,312 synapses at the
defaults, about 160,000 multiply-accumulates. A question takes one settle to hear, one or more
settles of the map and one settle a word said.
