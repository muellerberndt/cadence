# Eyes: a room seen as images

A parrot's eyes sit on the sides of its head. Each eye looks out sideways over a wide field, the
two fields overlap a little in front of the beak, and behind the head is a blind spot. The parrot
turns its head to look. `cadence.words.retina` gives a brain such eyes: it finds the things in
each eye's image, learns what they are and what colour they are from what it sees, and places
them on its map of things ([space](space.md)) from each eye's ray and the head's pose. Nothing
tells the brain what or where a thing is. The words brain ([words for things](words.md)) then
answers questions about what its eyes saw.

The module is opt-in: a program that does not import it gets a library without it.

| Piece | What it is | Section |
| --- | --- | --- |
| `Eye`, `parrot_eyes` | Where an eye's receptors look: sharp at its middle, coarse at its edges | [the eyes](#the-eyes) |
| `Retina.find` | The things in the images, where they are and how wide | [finding things](#finding-things) |
| `Categories` | Kinds and colours learned from what is seen | [what a thing is](#what-a-thing-is) |
| `Retina.see` | One moment of sight as the map's reports | [a first retina](#a-first-retina) |

## A first retina

The room is flat and seen across, so each eye's image is one row: a receptor a ray from the
eye, reporting the colour and the distance of what the ray meets. The world draws the images;
here a few lines draw round things before a round wall.

```python
import numpy as np
from cadence.space.map import Map
from cadence.words.retina import Retina


def image(eye, heading, things, wall=3.0):
    """One eye's image of round things (x, y, radius, rgb): each receptor's mean colour and
    distance over its rays."""
    origin = eye.origin(np.zeros(2), heading)
    angles = eye.facing(heading) + eye.rays
    u = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    depth = np.full(angles.shape, wall)
    rgb = np.full(angles.shape + (3,), 0.5)
    for x, y, radius, colour in things:
        w = np.array([x, y]) - origin
        along = u @ w
        miss = np.sum(w * w) - along ** 2
        t = along - np.sqrt(np.maximum(radius ** 2 - miss, 0.0))
        hit = (miss <= radius ** 2) & (along > 0) & (t < depth)
        depth = np.where(hit, t, depth)
        rgb = np.where(hit[..., None], np.array(colour), rgb)
    return np.concatenate([rgb.mean(axis=1), depth.mean(axis=1)[:, None]], axis=1)


retina = Retina()                                   # two parrot eyes, categories to learn
room = Map((-2.0, 2.0, -2.0, 2.0), (32, 32), units=retina.kinds.capacity,
           values={"colour": retina.colours.capacity})
red_ball, blue_box = (0.4, 1.1, 0.1, (0.9, 0.1, 0.1)), (0.4, -1.1, 0.05, (0.1, 0.2, 0.9))
heading = 0.0                                       # the beak points along x
images = np.stack([image(eye, heading, [red_ball, blue_box]) for eye in retina.eyes])
reports, lit = retina.see(images, (0.0, 0.0), heading, held=room)
room.moment(seen=reports, lit=lit)
for kind, place, values in reports:
    print(kind, np.round(place, 2), values)        # each thing's learned kind, place, colour
```

The left eye sees the red ball and the right eye the small blue one. Each enters a category of
its own: kind 0 and colour 0, kind 1 and colour 1. The map holds both, out of view as well
as in it, and a words brain (`cadence.words.Words(room)`) learns their names from naming.

## The eyes

`parrot_eyes()` gives two `Eye`s. Each looks out 70 degrees from the beak to its side and sees
80 degrees either side of that, so each sees from 10 degrees across the beak's line to 150
degrees back: 20 degrees in front are seen by both eyes and 60 degrees behind the head by
neither.

An eye has 321 receptors. Their spacing grows from 0.4 times the mean spacing at the eye's
middle (its fovea, 0.2 degrees) to 1.6 times it at the edges (0.8 degrees). A receptor pools
the light of six rays spread over its patch, so a receptor on a thing's edge reads a mix of the
thing and what lies behind it.

## Finding things

`Retina.find(images, at, heading)` gives each thing an eye finds as a `Figure`:

- **Edges.** A jump in distance between neighbouring receptors of more than `edge` (6 percent)
  of the nearer is an edge. A run of receptors between edges that lies nearer than the runs on
  both sides is a thing seen whole; nearer on one side only, a thing seen in part (another thing
  in front of it, or the field's edge).
- **Where an edge falls inside a receptor.** A receptor that mixes a thing and what lies behind
  it gives where the edge falls inside it by how far its distance lies between the two. A
  thing's own end receptor counts as such a mix where its distance lies more than `MIXED` (15
  percent) of the way to what lies behind. A thing's width is read within a few centimetres
  this way at every distance of the room (a disc's a little short: its edges lie where its sides
  curve away).
- **Where a thing is.** Each edge is a point on the plane, from the eye's place, its ray and the
  distance the ray reached. A thing's place is the middle of its edges, or, for a thing that
  bulges toward the eye, the point behind its front by its width times its bend over four (a
  disc's middle). A thing both eyes see is one thing (`fuse`).
- **How a thing bends.** The receptors' rays and distances are points on the plane; fitted by a
  parabola in the frame of the chord between the edges, their curvature times the width is the
  thing's bend: about 2 for a disc, 0 for a flat face, below 0 for a hollow open toward the eye.

A thing seen whole by at least `least` (16) receptors carries its form and its tint as
population codes: a unit tuned to a width and a bend together, and hue (at the colour's
saturation) beside lightness. The latest moment's things found and not recognised are
`Retina.unknown`; `Retina.look_at(place, at, heading)` gives the heading that brings a place to
the middle of an eye, where the fovea sees it sharply.

## What a thing is

`Categories(features, capacity)` learns categories from what is seen. Each category neuron is
joined to every feature neuron by a reciprocal pair of synapses that starts at zero:

- **Seeing** drives the feature neurons with a code; the category neurons settle and the most
  active is the category seen, where it settles above `vigilance`.
- **A lesson** drives the category's neuron and nudges the feature neurons toward the code (the
  library's free and nudged phases, a quadratic nudge). A category's n-th lesson closes 1/n of
  its error and no less than `rate`, so its synapses hold the mean of what it was seen as.
- **A new sort of thing** that settles no category above `vigilance` takes the next free one. One
  lesson's step closes about half its error, so a category enters imprinted: its first sight is
  taught `imprint` times and counts as one lesson.
- **What the map holds** at a thing's place is expected: `Retina.see(..., held=map)` takes the
  held kind and colour where the thing settles them above vigilance and within `EXPECT` of the
  most active, so one thing seen again stays one thing on the map.

The retina has two such layers, kinds over form and colours over tint. Their categories are the
map's kinds and colours: make the map with `units=` the kinds' capacity and `values={"colour":
...}` the colours'. Words name categories by naming, as they name a camera's kinds.

## What the brain does and what the program does

| | The program beside the brain | The brain |
| --- | --- | --- |
| The eyes' images | Drawn by the world (the chamber's renderer, or a page) | Read receptor by receptor |
| Edges, runs and mixes | Compares neighbouring receptors' distances (the eye's own wiring) | |
| Places and widths | Turns each edge's ray and distance into a point on the plane, as a body knows its head | |
| Form and tint | Codes the width, bend, hue and lightness as tuned units | |
| Kinds and colours | Drives the category layers with the codes and reads the most active | Settles on the category; its lessons hold what each was seen as |
| Looking | Turns the head to a thing it found and did not recognise | |
| The map | Passes the reports and the places the eyes see (`lit`) | Holds what is where ([space](space.md)) |

## Settings and cost

| Setting | Founder | What it sets |
| --- | --- | --- |
| `eyes` | `parrot_eyes()` | The eyes: 321 receptors each, axis 70 degrees, 80 either side, fovea 0.4, 6 rays a receptor |
| `kinds`, `colours` | 24, 8 | The category layers' capacities |
| `vigilance` | (0.4, 0.4) | The activity a code settles a category to, to be that category |
| `rate` | 0.02 | The least share of its error a category's lesson closes |
| `edge` | 0.06 | A jump in distance between neighbouring receptors, as a share of the nearer, that is an edge |
| `least` | 16 | The receptors that see a thing alone for its form to be read |
| `fuse` | 0.12 m | How near two eyes' things lie to be one thing |
| `near` | 0.15 m | How far short of a ray's reach a place lies to count as seen |

The parrot's retina has 4,864 learned parameters (24 kinds by 196 form units and 8 colours by
20 tint units, one efficacy a reciprocal pair). In a web page (Pyodide in headless Chromium, on a
box of the chamber's development runs), one moment of sight (finding, recognising with its
lessons, and the map's moment) takes 21 ms at the median and 33 ms at the 90th percentile, and
a question 24 ms.

The chamber is `benchmarks/words/retina.py`: the words chamber's script seen through the parrot's
eyes, beside the same brain fed the camera's reports, with its readings R1 to R6 (cadence#330).
