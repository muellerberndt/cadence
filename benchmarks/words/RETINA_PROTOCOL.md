# Eyes: declared protocol 1 (cadence#330)

Declared on 2026-11-13, committed to the lane branch before the first declared life. The founders,
arms, readings and gates below are fixed; a failed gate is reported as failed.

## The question

A parrot on its perch sees its room through two eyes on the sides of its head
(`cadence.words.retina`): each eye's image is a row of receptors, each the colour and the distance
of what its ray meets. Nothing tells the brain what or where a thing is. Does it find every thing
in view (R1), hold the things' places on its map as the camera's reports do (R2), name the kind
and the colour of things at distances and turns no room holds (R3), answer the words chamber's
questions as well as the same brain fed the camera's reports (R4), beside a convolutional network
and a vision transformer given the same images and episodes (R5), and run in a web page at the
page's frame rate (R6)?

## The brain

At the declared commit, with its defaults:

- **The retina** (`Retina()`): two parrot eyes (`parrot_eyes()`: 321 receptors each, looking out
  70 degrees from the beak and 80 either side, fovea 0.4, 6 rays a receptor); edges at a jump of
  6 percent; a thing's form read at 16 receptors or more; two eyes' things within 0.12 m are one;
  a place seen where a ray reaches 0.15 m past it. Kinds: 24 categories over a joint code of width
  (14 units) and bend (14 units) at vigilance .40; colours: 8 categories over hue (12 units) and
  lightness (8 units) at vigilance .40; a lesson closes 1/n of its error, no less than .02; a
  category enters imprinted (4 lessons); what the map holds at a place is expected there.
- **The map**: `Map((-2, 2, -2, 2), (32, 32), units=24, values={"colour": 8})`.
- **The words brain**: `cadence.words.Words` on that map with its defaults, practice told
  (protocol 1 of #326).
- **Looking**: after looking around a room, the parrot turns its head to each thing it found and
  did not recognise (6 at most), one moment each, and back to the room's heading.

Learned parameters: 15,104 (the lexicon's 10,240 pairs, the kinds' 4,704 and the colours' 160).

## The chamber

`benchmarks/words/retina.py`. The script is the words chamber's protocol 1 (`words.script`, its
defaults), with each room drawn again until no thing hides any part of another from the perch
(1.5 degrees apart) and no thing stands 1.25 to 1.35 m from it (R3's band). Each kind is an
outline on the floor (disc, oval, board, dish) in one of three sizes (0.12, 0.19, 0.30 m); the
four new kinds are combinations no trained kind has. A thing faces the perch, turned up to 15
degrees; it is drawn in its colour, lit from the parrot's side. Receptor noise: 0.1 percent of the
distance, 0.01 on each colour.

## Founders and arms

Founders 33000000 to 33000011 (twelve, no life in any lane). R6's timing founder 33099999.

| Arm | What it is | Parameters |
| --- | --- | --- |
| eyes | the brain above | 15,104 |
| camera | the words chamber's brain (`words.make_brain`), fed the camera's reports of every thing whose middle lies in either eye's field, on the same script | 6,656 |
| eyes, camera by pistachios | both arms with practice by reward (protocol 2 of #326, 60 rooms), a reading | 15,104; 6,656 |
| CNN, ViT, online and offline | `retina_controls.py`: the brain's images and episodes, the brain's looks included | at 15,104 (nearest width) |
| CNN, ViT, offline, large | the same at nine times it, a reading | at 135,936 |

The controls see what the parrot's eyes see, image for image (the same receptor noise in the same
order), and where the head faced; each naming as the teacher's pointer and the heard word; each
question as words; the training and the modes are the words chamber's controls'.

## Readings

- **R1**: at every moment of the test rooms, each thing wholly inside an eye's field is found
  where a figure lies within 0.15 m of its place; the share of figures that lie near no thing.
- **R2**: at each test room's first question, each thing's place on the map from the eyes (the
  held thing nearest it) against its place on the map from the camera's reports (its kind's
  nearest): the share within 0.10 m, for things in either eye's field and out of both.
- **R3**: 96 views a founder of one trained thing in a drawn colour, 0.45 to 0.6 m away or in the
  band 1.25 to 1.35 m, turned 20 to 30 degrees, the parrot looking at it (within 3 degrees of an
  eye's middle); the map emptied, one moment of sight with no lesson, then "what is <place> you?"
  (its kind's word) and "what colour is the <kind>?" (its colour's word). Read: both right.
- **R4**: the words chamber's W1 and its named-only part, W2, W3, W4 never seen and never
  taught, W5 by colour and by place, W4 out of view against the map's own rate, W6's ablation;
  for the eyes and the camera arms.
- **R5**: the table of every arm with parameters and multiply-accumulates.
- **R6** (`retina_page/run.py`): in headless Chromium, Pyodide 314.0.7 with the wheel of the
  declared commit: a short life of founder 33099999 (4 naming rooms), each moment of a turn timed:
  the brain's moment (finding, recognising with its lessons, the map's moment), drawing the
  images, a question.
- Also: the categories formed and their purity, the looks.

## Gates

A gate holds where, for each of its readings, the mean over the twelve founders is on the stated
side and at least ten of twelve founders are.

- **R1, things found**: found at least .95; figures that are no thing at most .05.
- **R2, the map from the retina**: in view at least .95; out of view at least .95.
- **R3, what a thing is, from the image**: kind and colour named at least .90.
- **R4, the questions on the retina**: for each of W1, W1 named-only, W2, W3, W4 never seen, W4
  never taught, W5 by colour and W5 by place, the eyes arm less the camera arm at least -.05 in
  the mean and at least -.10 on ten founders; W4 out of view within .05 of the eyes map's own rate;
  W6: with the lexicon's synapses at zero, W1 at most .05 on every founder.
- **R5, beside the controls**: the table, every arm with parameters and multiply-accumulates.
- **R6, in the browser**: the brain's moment at most 33 ms at the median (30 frames a second);
  a question at most 100 ms at the median.

**Expectations, stated before the run.** From development founders 980 to 985 and 990 to 999 (no
gated life), at the declared settings: R1 found 1.00, figures that are no thing .008; R2 1.00 in
view and out; R3 1.00; the eyes arm reads as the camera arm on every reading of R4 (mean
difference .000; W1 1.000, W2 1.000, W5 by colour .994, by place .999); one founder of sixteen
reads W3 at .69 in both arms (the word for white settles on a kind both arms named it with); 12
kind categories for 12 kinds at purity 1.000. R6 in development: 21 ms a moment, 24 ms a question.
The controls' readings are unknown before the run (the words chamber's controls read W1 .06 to
.21).

**What a failure names.** An R1 miss names the edges; an R2 miss the places from the rays; an R3
or R4 miss with a category shared by two kinds the codes or the vigilance, with a word on the
wrong category the naming of a thing seen as two.

## Compute

One AWS c7i.8xlarge box (Name=cadence-0100-words-retina), one process a life, one BLAS thread a
process, results synced to s3://cadence-0100-words-retina-20261113 (`retina_gates.py run`, then
`retina_page/run.py` with nothing else running).
