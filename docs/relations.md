# Relations between things

A parrot that holds a map of its room learns words for where things are from each other and
from itself: in front of, to the right of, behind, to the left of, near, far from. It learns
them from questions answered for it, and then answers in both directions: "where is the ball
from the cup?" (from two things to a word), "what is left of the ball?" (from a word to the
things), "is the cup near the apple?" (yes or no), and the same from itself: "what is behind
you?".

`cadence.words.relations.Relations` is a part of a [`Words`](words.md) brain
([parts that plug in](words.md#parts-that-plug-in)): `Words(map, parts=[Relations()])`. It is
opt-in: a program that does not import it gets a library without it.

## A first parrot that learns relations

```python
import math
import numpy as np
from cadence.space.map import Map
from cadence.words import Words
from cadence.words.relations import Relations

kinds = ["dako", "mipu", "sefa", "lobi"]               # nonce words: ball, cup, box, book
quarters = ["fobe", "lume", "kosa", "riza"]            # in front of, left of, behind, right of
relations = Relations()
parrot = Words(Map((-2.0, 2.0, -2.0, 2.0), (24, 24), units=4), parts=[relations],
               capacity=32, shares={"kind": 0.4, "place": 0.3},
               heard_only=True, pattern_back=True)        # the lexicon's opt-ins, below
rng = np.random.default_rng(0)


def new_room(right=None):
    """Four things around the parrot, one in each quarter of where it faces (the one to its
    right at ``right`` metres, where given)."""
    facing = rng.uniform(-math.pi, math.pi)
    places = [rng.uniform(0.7, 1.7) * np.array([math.cos(a), math.sin(a)])
              for a in facing + np.arange(4) * math.pi / 2 + rng.uniform(-0.5, 0.5, 4)]
    if right is not None:
        places[3] *= right / np.linalg.norm(places[3])
    parrot.map.reset()
    parrot.map.moment(seen=[(k, places[k]) for k in range(4)])
    parrot.body.heading = facing
    return places


for _ in range(12):                                    # naming: the things
    places = new_room()
    for k in range(4):
        parrot.name(kinds[k], places[k])
relations.name_self(parrot, "you")                     # and the parrot itself
for _ in range(48):                                    # questions answered: where from you?
    new_room()
    for k in range(3):                                 # three of the four quarter words
        parrot.practice(f"where is the {kinds[k]} from you", quarters[k])
for _ in range(4):                                     # what is in front of you?
    new_room()
    parrot.practice(f"what is {quarters[0]} you", kinds[0])

new_room()                                             # a room it has never seen
assert parrot.ask("where is the sefa from you") == ["kosa"]
assert parrot.ask("what is kosa you") == ["sefa"]      # never asked this way about "kosa"
```

The parrot was never told which word names a direction: each answer word was named with what
the parrot sensed of the thing from itself, its direction and its distance, and the word kept
what its lessons shared, the direction.

**The lexicon's two opt-ins.** The vector cells are population codes: neighbouring codes
overlap, and so do the patterns of the words for them. `Words(heard_only=True)` makes a lesson
move the synapses of the words heard alone (by default every word's synapses of the taught
groups step, and a word active through its overlap with the heard word is taught toward
silence). `Words(pattern_back=True)` makes the motor hear a word back at the level of the group
of a concept said as a pattern (by default at its exact unit, which a population code's peak
rarely shares with the right word's). In development lives of the chamber (no declared run),
"where" and "how far" answers read .900 (from things) and .862 (from the parrot) with neither
(six founders), and 1.000 and 1.000 with both (sixteen founders).

## A word taught through words the parrot knows

"What is next to the apple?" answered with the cup: where a question holds one word the parrot
hears for the first time, every other word known and no relation word among them, the new word
takes the meaning the answer shows. Each answered thing's code from the reference names it,
and from then on it is a relation word like the others. A teacher shows it where it varies in
what it does not name (here the distance), so that a few examples tell what it names.

```python
assert parrot.lexicon.unit("riza") is None             # the fourth quarter word: never heard
for right in (0.7, 1.0, 1.4, 1.8):                     # shown at four distances
    new_room(right)
    parrot.practice("what is riza you", kinds[3])      # answered with the thing to the right
new_room()
assert parrot.ask("what is riza you") == ["lobi"]
assert parrot.ask("where is the lobi from you") == ["riza"]   # and said the other way
```

## What the parrot is born with and what it learns

| | Born with (the program, or a set value) | Learned |
| --- | --- | --- |
| Where a thing is from a reference | Vector cells: 8 direction cells, each a von Mises curve around the turn, in the frame of where the parrot faces, and 6 distance cells on a log scale from 0.3 m to 3 m | Nothing: the cells are a sense |
| What a relation word means | | The pattern its lessons leave on the cells, in the lexicon's synapses: a direction word's on the direction cells, "near" and "far" on the distance cells |
| Where "left" ends, how near is near | | Where another word's pattern wins the motor's choice |
| Which cells a word is about | | The group where the word's pattern is sharpest: its lessons agree there and vary in the other group, which its next lessons teach toward zero |
| "you" | | A word named while the parrot attends to itself |
| Which of two things is the reference | The parrot is not on its own map, so it is never the thing placed | Of two things, the first or the last named (the anchor read), learned where an answer decides it |
| "yes" and "no" | | Words named with the parrot's own verdict |

No table maps a relation word to a test of where things are. With the synapses between the
words and the vector cells at zero, the parrot says no relation and finds no thing by a relation.

## How a question is answered

1. **The words, one by one.** Each word of the question is heard alone. A word that settles on
   a kind names a thing; one that settles on the parrot itself names the parrot; a relation
   word is a word the parrot says back for its own hearing on the vector cells.
2. **The reference.** One thing named is the reference ("what is left of the ball?"). Of two,
   the parrot is never the thing placed ("where is the ball from you?"), and of two things the
   anchor read says which ("where is the ball from the cup?").
3. **Placing.** The map settles on the reference's place and the thing's, and the vector cells
   give the thing's code from the reference.
4. **Describing.** For each group of cells, the code drives the lexicon's concepts, the word
   neurons settle, and the motor takes the most active word whose own hearing settles on that
   group: one word for the direction, one for the distance.
5. **The answer.**
   - "What is left of the ball?": every other thing on the map, found one at a time by the
     map's settle, is described from the ball; the things whose description holds a relation
     word of the question are found, and the question's read says what is said of them (their
     kind).
   - "Where is the ball from the cup?": the ball is described from the cup, and the read says
     which group is said (the direction for "where", the distance for "how far").
   - "Is the cup near the apple?": the cup is found with whether its description from the
     apple holds "near" (`truth`), and the read says the word for that verdict.

## How a relation word is learned

- **"Where is the X from the Y?" answered with a word.** The answer word is named with the X's
  code from the Y, each group at a gain: how sharply the word's own hearing settles there (its
  peak above its mean). A group where its lessons' codes agree stays sharp; one where they vary
  becomes a broad mean, loses its gain, and its synapses are taught toward zero. Where the
  answer is said for one order of the two things and not the other, the question's words are
  taught that order (the anchor read).
- **"What is left of the X?" answered with things.** The question's relation word is named with
  each answered thing's code from the X.
- **"Is the X left of the Y?" answered with a word.** The word is named with the parrot's own
  verdict.
- **Synaptic scaling.** After each lesson a relation word's synapses onto the vector cells are
  held at one length (`norm`), so the motor's choice turns on the shape of a word's pattern,
  whatever its size: a word taught in four examples meets words taught in fifty on equal terms.

## Settings and cost

| Setting | Default | What it sets |
| --- | --- | --- |
| `directions`, `sharpness` | 8, 2.5 | Direction cells and the von Mises sharpness of each |
| `distances`, `nearest`, `farthest`, `width` | 6, 0.3 m, 3.0 m, 0.37 | Distance cells, their range on a log scale, and each one's width in natural log units |
| `tries` | 64 | The most active words the motor hears back, in order, for one of the group it says |
| `norm` | 1.2 | The length a relation word's synapses onto the vector cells are held at; None leaves them as the lessons set them |
| `contrast` | 1.0 | The power of a group's sharpness in a word's gain there |

The part adds 19 concepts to the lexicon (8 direction cells, 6 distance cells, the parrot, the
two verdicts, the two anchors) and three reads (direction, distance, the verdict), so a lexicon
of `capacity` words holds `capacity` times 19 more learned parameters. A question with a
relation takes one settle of the lexicon a word heard, one a word said back, two a thing
described, and the map's settles that find the things.

## The chamber

[`benchmarks/words/relations.py`](../benchmarks/words/relations.py) raises a parrot over rooms
of nonce words drawn for each founder: naming, questions answered from the parrot and from
things, a relation word held out of practice and taught in four answered questions, and a test
of every form in fresh rooms, beside an LSTM and a transformer given the same script
([`relations_controls.py`](../benchmarks/words/relations_controls.py)).

The declared run ([protocol R1](../benchmarks/words/RELATIONS_PROTOCOL.md)), twelve fresh
founders, read at 2599c700 (means over the founders):

| Arm | Parameters | Things: from a thing / the parrot | Where, how far: thing / parrot | Is it: thing / parrot | Held-out pairs | A word taught in four answered questions |
| --- | --- | --- | --- | --- | --- | --- |
| the parrot | 2,368 | .985 / .981 | 1.000 / 1.000 | .997 / .994 | .992 | .974 |
| LSTM, offline | 2,392 | .174 / .086 | .576 / .537 | .474 / .457 | .344 | .128 |
| transformer, offline | 2,424 | .157 / .063 | .513 / .501 | .472 / .464 | .310 | .177 |
| LSTM, offline, 9x | 21,400 | .247 / .083 | .593 / .557 | .599 / .589 | .376 | .225 |
| transformer, offline, 9x | 21,768 | .183 / .080 | .568 / .531 | .535 / .467 | .334 | .220 |

With the synapses between its words and the vector cells at zero, the parrot answers none of
the questions about things or relations whose answer is a thing or a relation word (.000).
