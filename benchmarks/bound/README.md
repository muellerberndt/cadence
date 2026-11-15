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
  and a surface and no place.
- **Back into the brain.** The word returns to the ear and the thing to the glimpse, wherever
  the eyes report nothing, and the brain settles on it.
- **The reading** is the completion matrix: one facet as the cue, the facets it implies read
  from the association's settled state by linear probes fitted on looks at whole things.

Three reads of the state are compared as the key: the association region's activity, the
sparse mixed code where the attended thing's senses and the heard word meet at unit length, and
the two side by side.

## A named thing draws attention

`named.py` joins the two. At every heard word the records are read by the ear alone. Where the
heads agree on a form, the recalled thing is looked for among the things the map holds, and the
focus moves to the thing that matches: from the next moment the glimpse is the map's read of
it. The rule is born in the instrument: which read holds to the ear, and that a recalled form
moves the focus, are set by hand. A learned gaze is the open step.

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
