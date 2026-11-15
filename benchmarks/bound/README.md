# Bound chamber: one attended thing, held by the map and completed from a part

This chamber holds three instruments for the composite brain of
[#342](https://github.com/muellerberndt/cadence/issues/342). They read the brain of the meaning
chamber ([#334](https://github.com/muellerberndt/cadence/issues/334)), which answers questions
about the thing it attends, and ask three things of it:

| Instrument | Question |
| --- | --- |
| `mapped.py`, `life.py`, `occluded.py` | Can the attended thing reach the brain from the map of things (`cadence.space.map`), so that a thing out of view is answered about as a thing in view is? |
| `completion.py` | Does one facet alone (a word heard with nothing attended, one sense of a thing) bring the whole thing back into the brain's settled state, through records in one `cadence.longterm.LongTermMemory`? |
| `named.py` | Does a heard name move the focus to the thing it names, so that "what colour has the X" is answered about X while the trainers attend another thing? |

It is a research chamber on one founder. It establishes no default and declares no protocol.
The meaning chamber and its senses live on the branch `lane-0100-meaning`; `lane.py` reads them
from a checkout of that branch (`CADENCE_MEANING_LANE`, or `../wt-meaning` beside the
repository). The library is this branch's.

## The map as the source of the glimpse

In the meaning chamber the thing in focus is reported twice by the eyes: in a slot of the scene
and in a place of its own, the glimpse. Here the glimpse is read from the map.

- **The eyes write the map.** Every thing in view is reported with its place around the body
  and with each sense's tuned cells (hue, lightness, shape, size, surface) as codes carried on
  its cells (`Map(values=)`, a value as a code).
- **The brain reads the thing it attends.** `Map.thing_at` settles on the cells that hold the
  attended thing and gives its codes as the map holds them and its place from the body: the
  bearing from where the head points and the distance, which the eyes' own tuned cells code. The
  read enters the brain where the eyes' direct report of the thing in focus entered.
- **Nothing else changes.** The scene's slots stay the eyes' report of what is in view. The
  brain's settle, its lessons and its voice are the meaning chamber's.

The map takes a kind's number with each report, which tells it which report is which thing.
Here that number is the world's kind of the thing. The brain reads no kind unit: it reads the
values and the place. The focus is given as the kind of the thing the trainers attend, and
where that thing is comes from the map.

`life.py` lives the founder's life from its checkpoint after imitation with this glimpse
(names, colour, where, then the pretraining mix), with the draws of the equal-senses brain's
life. `occluded.py` reads a brain on things in view, behind a screen, and moved while behind
it, with the glimpse from the eyes and from the map.

## Completion from one facet

`completion.py` keeps records of whole bound things in one `LongTermMemory` and reads them by
a part.

- **A record** is written when a trainer names a thing the brain attends. Its value is the
  bound thing: the glimpse's sense populations and the heard word. Its key is a read of the
  brain's settled state at that moment.
- **A read** takes the same read of the state at a later moment as its cue. Several heads read
  one record each (`LongTermMemory.read(cue, heads=)`), and a population of the bound thing
  comes back where every head's record holds it alike. "Apple" heard alone brings back a form
  and a surface and no place. A read whose records hold a population otherwise than the cue
  itself holds it (another word than the one heard) returns nothing.
- **Back into the brain.** The word returns to the ear, and the thing returns as a thing in
  focus is reported: to the glimpse, wherever the eyes report nothing, and to an empty slot of
  the scene. The brain settles on it.
- **The reading** is the completion matrix: one facet as the cue, the facets it implies read
  from the association's settled state by linear probes fitted on looks at whole things.

Three reads of the state are compared as the key: the association region's activity, the
sparse mixed code where the attended thing's senses and the heard word meet at unit length, and
the two side by side.

## A named thing draws attention

`named.py` joins the two. At every heard word the records are read by the ear alone. Where the
heads agree on a form and the word that comes back is the word heard, the recalled thing is
looked for among the things the map holds, and the focus moves to the thing that matches: from
that moment the glimpse is the map's read of it. The rule is born in the instrument: which read
holds to the ear, and that a recalled form moves the focus, are set by hand. A learned gaze is
the open step. Where the map holds no matching thing, the recalled thing returns to the senses
as `completion.py` returns it, and the brain answers about a thing that is nowhere in view.

## Readings

Development readings on the meaning chamber's founder 33400121, every one frozen and greedy.
Unless a section says otherwise the brain is the meaning chamber's equal-senses brain: it was
trained with its glimpse from the eyes and is read here with no further lesson.

### A thing out of view

`occluded.py`, 96 questions a row; the colour forms ask kind-colour pairs kept out of training.

| The thing asked about | Glimpse from | name | colour_of | colour_this | material_this | direction | is_round | can_eat_bal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| in view | the eyes | .97 | .71 | .75 | .77 | .91 | .93 | .99 |
| in view | the map | .97 | .71 | .75 | .77 | .91 | .93 | .99 |
| behind a screen | the eyes | .00 | .04 | .02 | .25 | .24 | .00 | .00 |
| behind a screen | the map | .98 | .78 | .78 | .76 | .92 | .94 | .99 |
| behind a screen, moved by a sound | the eyes | | .01 | | | .26 | | |
| behind a screen, moved by a sound | the map | | .69 | | | .93 | | |

- The map's read of a thing lies within 0.001 of the eyes' report of it on every input, and the
  brain's sparse mixed code of the two is the same code. A brain trained on the eyes' glimpse
  reads a thing in view alike through the map.
- A thing behind a screen is answered about at the rates of a thing in view. Through the eyes
  nothing of it reaches the brain.
- A thing that moved while hidden is answered at its new place, with its colour.
- `compare.py`: the demo room (.80 right in all three passes), fresh rooms (.93 on pairs of the
  training rooms, .81 on kept-out pairs) and every focus form read the same through the map as
  through the eyes.
- The demo room through the page's host (`mapped.MappedHost`), every question on every thing in
  one pass: .84 with the thing in view, through the eyes and through the map; with the thing
  asked about behind a screen, .06 through the eyes and .86 through the map.
- The map's own read regions follow the attended thing: `where` and `what` move by .85 of their
  length from the settle of a look to the settle of the attended read and by .66 between the
  reads of two things of one room, and `where` lies on the attended thing's cells alone. The
  carried sheet is the same after a read.

### A brain trained with its glimpse from the map

`life.py` lived the pretraining mix (400 blocks, lesson weights .25, .10, .05) from the
equal-senses lineage's checkpoint after names, colour and where, with its glimpse read from the
map at every moment, the draws of the equal-senses brain's own mix. Read at its last block:

| Reading | Equal-senses brain (glimpse from the eyes) | Trained through the map |
| --- | --- | --- |
| demo room, right in all three passes | .80 | .84 |
| fresh rooms, pairs of the training rooms | .93 | .93 |
| fresh rooms, kept-out pairs | .81 | .83 |
| "what colour has the X" on kept-out pairs (chamber) | .77 | .84 |
| "what colour is this" on kept-out pairs (chamber) | .74 | .82 |
| demo room through the host, the thing asked about behind a screen | .86 | .90 (.08 through the eyes) |
| "where is the X", the thing behind a screen and moved (chamber) | .93 | .96 |

The brain trained through the map reads the same through the eyes as through the map on every
row with things in view: in view the two glimpses are one code.

### Completion from one facet

`completion.py`: 400 things, each named by every word that names it; five heads. A cell holds
the share read right from the association's settled state one moment and two moments after the
cue, by probes fitted on looks at whole things in the meaning chamber's rooms of three to five
things. A whole thing in focus reads kind .77, colour .90, round .99, material .80, edible .93,
direction .90 and distance .84 by those probes.

**The key.** For a word heard with nothing attended, the share of the heads' records that are
named by that word:

| Key | a kind's word | a colour's | a material's | a direction's |
| --- | --- | --- | --- | --- |
| the association's state | .19 | .11 | .29 | .12 |
| the sparse mixed code | 1.00 | 1.00 | 1.00 | 1.00 |
| both, each at unit length | 1.00 | 1.00 | 1.00 | 1.00 |

The association's code of a thing named and its code of the name heard alone lie too far apart
for a read by cosine: a part is no part of that code. In the sparse mixed code each sense and
the heard word enter at unit length, and a word alone shares about half of the code of a thing
named by it.

**A kind's word heard alone, nothing attended** (key: the sparse mixed code):

| Room | Arm | kind | round | material | edible |
| --- | --- | --- | --- | --- | --- |
| empty | no records | .11, .00 | .56, .56 | .00, .00 | .56, .56 |
| empty | the records | .89, .89 | .89, 1.00 | 1.00, 1.00 | .89, .78 |
| empty | the records, values dealt to other keys | .00, .00 | .44, .33 | .00, .00 | .44, .33 |
| other things | no records | .01, .01 | .37, .41 | .00, .00 | .66, .59 |
| other things | the records | .61, .80 | .72, .88 | .24, .56 | .88, .98 |
| other things | the records, one head | .71, .89 | .92, .96 | .27, .65 | .88, .96 |
| other things | the records, key the association's state | .06, .05 | .36, .36 | .00, .00 | .67, .67 |
| other things | the records, values dealt to other keys | .01, .02 | .33, .33 | .00, .00 | .67, .67 |

The empty room holds one cue a word. For a kind's word the heads agree on every shape line and
the size at every read, on the surface where the kind has one material, on the hue for a third
of the kinds and on no direction.

**The other words**, in a room of other things, two moments after: a colour's word reads its
colour at .92 (.15 with no records, .16 in the control), a direction's word its direction at
.75 (.14, .24), a material's word its material at .51 (.25, .23).

**What each part of the return does** (a kind's word, the kind two moments after). In a room of
other things: the word alone .04, the thing alone .49, the thing in the glimpse with the word
and no slot of the scene .73, all of it .80. In an empty room: without the slot of the scene
.11, with it .89. This brain's state reads a thing only with a scene around it: a thing shown
to the eyes in the glimpse alone, with every slot of the scene empty, reads its kind at .17.

**One sense of a thing alone.** With only the colour in the glimpse the colour reads .27, and
.68 after the read; the control reads .69, since any return that makes the percept whole lets
the glimpsed colour be read. The surface alone reads its material at .20, .50 after the read
and .26 in the control. The form alone reads its kind at .76 with no records.

**A thing seen, nothing heard.** A word comes back in .02 of the reads with five heads and in
.07 with one: the nearest records of a seen thing hold each of the words that name it and
things of other colours and places. Which facet's word is wanted is nothing this read says.

### A named thing

`named.py`, 96 questions a row in rooms of three things; the share of replies that give the
asked thing's answer, and in brackets the share that give the focus thing's.

| Question | The trainers' focus | The asked thing | Rule off | Rule on |
| --- | --- | --- | --- | --- |
| what colour has the X | on the asked thing | in view | .97 | .97 |
| what colour has the X | on another thing | in view | .00 (.99) | .91 (.04) |
| what colour has the X | on another thing | behind a screen | .00 (.99) | .89 (.08) |
| what colour has the X | on nothing | in view | .14 | .92 |
| what colour has the X | on nothing | behind a screen | .01 | .92 |
| where is the X | on the asked thing | in view | .93 | .93 |
| where is the X | on another thing | in view | .06 (.85) | .85 (.08) |
| where is the X | on another thing | behind a screen | .06 (.88) | .90 (.06) |
| where is the X | on nothing | in view | .28 | .91 |
| where is the X | on nothing | behind a screen | .27 | .93 |

With the rule on, the name draws the focus to the asked thing in every exchange and to no other
thing. The forms that say no name read the same with the rule on as off (the focus is drawn in
none of their exchanges). The focus moves at the moment the name is heard, one moment before
the brain's turn, so the brain has looked at the named thing once when it answers.

**A named kind that is nowhere in the room.** Where the map holds no thing that matches, the
recalled thing itself returns to the glimpse and to a slot of the scene, and the voice answers
from it. "What shape has the X" of a kind no thing in the room is of, the focus on nothing, 96
questions (the right answer is "round" for a round kind and "no" for another):

| Arm | round kinds | other kinds |
| --- | --- | --- |
| rule off | .00 | .05 |
| rule on | .97 | .98 |
| rule on, another kind's records recalled in its place | .03 | .24 |

The answer follows the thing the name recalls: with the records of a kind of the other shape
class recalled, the voice answers for that kind.

### Limits

- One founder; development readings with no declared protocol. The brain trained through the
  map lived the pretraining mix through the map from a checkpoint whose names, colour and where
  were learned through the eyes.
- The records' key that completes a word is the sparse mixed code, the population where the
  attended thing's senses and the heard word meet before the settle. Keyed by the association's
  settled activity, a word alone finds records named by it on .11 to .29 of the heads' reads.
- A read is a prototype of the things a word named. With records of fewer things it is
  coarser: on records of 80 things (350 namings) a word alone returns its own word on .97 of
  the words; on records of 40 things, .69. At 160 things a ball's name drew the focus to an
  apple in one room of 32.
- Five heads that must agree say nothing of what the named things differ in; one head returns
  one record whole. In a room of other things one head reads a kind's word at .89, five heads
  at .80.
- A sense alone gains little over its control, and a thing seen with nothing heard returns no
  word: which facet's word is wanted is nothing the read says.
- Which read holds to the ear, and that a recalled form moves the focus, are born rules of
  `named.py`, set by hand. The map takes the world's kind number with each report.

## Run

```sh
git worktree add ../wt-meaning origin/lane-0100-meaning
python benchmarks/bound/life.py <N0 brain.npz> <out dir>            # the life with the map
python benchmarks/bound/occluded.py <brain folder> occluded.json
python benchmarks/bound/completion.py <brain folder> completion.json
python benchmarks/bound/completion_table.py completion.json
python benchmarks/bound/named.py <brain folder> named.json
python -m pytest benchmarks/bound -q
```

`<brain folder>` holds a `brain.npz` of the meaning chamber's founder 33400121.
