# The text parrot: declared protocol 1 (gates G1 to G6)

The owner's request: a parrot that learns everything about the things in its world (colour,
shape, material, size, food, where they are) and the grammar to speak about them, with text in and
text out; it answers questions such as "what is round?", says a scene in sentences, and is then
an expert at this, its understanding read from inside its brain.

## The chamber

- **Things.** Twelve kinds; ten are shown in pretraining, two (a lemon and a pen) never are. Six
  colours, three shapes, six materials, two sizes; fruit is food. Eight places: four directions
  around the parrot at two distances. The eyes report hue, saturation, lightness, outline,
  width, surface, four details and the direction and distance of the thing in view, with noise;
  no kind, colour, material or place is reported as such (`textworld.py`).
- **Forms.** Seventeen question forms with a depth knob ("the apple", "the red apple"), among
  them the owner's: "what colour has the apple ?" ("red"), "what is left of the apple ?"
  ("block"), "where is the apple ?" ("left far"), "where is the cup ?" ("behind"), "what objects
  are round ?" ("apple cup"). Sentence replies: "what do you see ?" (one clause a thing, "the red
  apple is left far") and "what do you see left of the apple ?" ("the blue block is left of the
  red apple"); a parser reads a reply back into the things, colours, places and relations it
  says.
- **An exchange** (`textchamber.py`). The question one word a moment; the gap after it closes the
  held message; the eyes sweep over the scene (a born serial scan, below) and the parrot's gaze
  holds the thing the question names (learned); the eyes sweep again and at each thing the voice
  says words, one a moment, each heard back at the next; its silence moves the eyes on; after the
  sweep it says ".". The voice is held while the teacher speaks; those moments teach nothing.
- **Every learner** (the brain and each control) is driven through the same moments: the same
  input vector a moment (the word read, the gap, the glimpse, the focus) and the same two acts
  (voice, gaze).

### The born parts and their learned replacements

| Part | Born | Evidence | The learned replacement |
| --- | --- | --- | --- |
| The serial sweep over the things | A fixed order (left to right, then behind), one thing a moment, every thing visited whatever the question | Serial search over conjunctions of features (feature integration), inhibition of return; birds and parrots scan a scene by head saccades | A gaze slot that chooses the next fixation from a coarse map of the scene, taught by whether the reply was right |
| The focus (a held thing stays in view) | The gaze's hold keeps that thing's glimpse in a place of its own | Object files and visual indexes; the tectum and pallium keep the selected stimulus | A learned working trace of the held thing |
| The voice held while the teacher speaks | No lesson and no heard act in the question's moments | The songbird's song nuclei are inhibited while it listens | A learned turn: the gap as the cue to speak (the meaning lane's turn rungs) |

The sweep never decides an answer: it visits every thing in its fixed order whatever the
question, and what is said at a thing, and whether anything is, is the brain's.

## The brain

`Brain.compose` (`textparrot.py`): a chain of processing regions ending in the association region
(whose settled activity is the concept code), the apical compartment carrying each region's
input from the region above (top-down), the held message holding the question from the gap on,
the heard history (decays 0 and .6), the working trace at .3, two motor slots (voice: every word
and silence; gaze: pass or hold), at `teaching="deep"`, lessons only where the answer differs
(`learning_surprise=True`, amendment 1). The words read reach the
lowest region with the eyes, or (`ear` arms) the association region alone, so the question reaches
the regions below only through their apical compartments. Sleep (`s` arms): every efficacy times
.5 between stages.

**Pretraining** by teacher forcing and DAgger (from the 21st block of a stage, three blocks in ten:
the parrot's own words heard and moving its eyes, each moment labelled with the teacher's word from
where its own words left the exchange) through six stages: words (one thing: what is it, what
colour, what shape, the echo), properties (what is red, what objects are round, small, made of
wood, what can you eat, is a key long, can you eat a banana), relations (where, what is left of),
questions (what colour has, can you eat the), sentences (what do you see, one or two things),
scenes (what do you see, what do you see left of). Three blocks in ten revisit the earlier
stages' forms. A third of the echoes are nonce words (heard and said back, never a name). A stage ends at mastery (a frozen greedy read of its forms at .90, every 20 blocks)
or its cap (words 200, properties 600, relations 600, questions 400, sentences 300, scenes 600
blocks of 64 exchanges; amendment 3).

## The arm

The development sweep (founders 900 and 901, caps of 150, 300, 300, 200, 200, 300) reads every
depth (2, 4, 6 regions) at every apical fraction (0, .25, .5), the `ear` arms, sleep and
every-row lessons (`d2a25all`). **The gated arm is the deep apical arm** (four or six regions, apical
above zero) with the best development mean on G3, unless an arm of two regions or without the
compartment is above every deep apical arm by more than .05 on G3; then that arm is gated beside
the best deep apical arm and the report reads why depth did not pay.

## The gated run

Twelve fresh founders, 33800000 to 33800011, each: the brain's life; the controls below; the
readings. Fresh scenes for every reading (the founder's reading stream, never a training scene).

| Gate | Reading | Passes at |
| --- | --- | --- |
| G1 | The exam forms on fresh scenes, the reply word for word: colour, relation, where, which shape, and the mean of all eleven question forms | each at .90 (mean over founders) |
| G2 | "what do you see ?" and "what do you see R the X ?": the parser reads the reply back to the true scene | each at .90 |
| G3 | Held out: colour x kind pairs never shown (colour, where, which colour, describe), relation x referent pairs never asked, forms at a depth never taught (where, can eat), new objects (the probes on lemons and pens) | pairs .90, relations .85, depth .90, new objects .85 |
| G4 | Introspection: (a) probes round, red, food, left, near on fresh things; (b) a kind word alone: balanced accuracy of round, food and red from the seeing probes, and the same thing (hearing code nearest its kind's seeing code); (c) silencing the round, red and food directions; (d) a changed scene; (e) neighbours of the new kinds; (f) the regions | (a) each .90; (b) .80 and .50 (chance .10), above the newborn; (c) the property's form falls by .5 or more, the spared forms by .10 or less; (d) .90; (e) the lemon nearest a round or fruit kind and the pen nearest a long kind on 10 of 12; (f) read, no gate |
| G5 | A new word ("zabu") for a new thing (the lemon) by pistachios alone: odd trials the trainer says the word, even trials "what is it ?", a pistachio for "zabu", the rewarded exchange's own acts taught; a frozen read on 16 fresh lemons every two trials | .90 within 40 trials on 10 of 12; then round and food under the new name at .80 |
| G6 | Matched controls: LSTM and transformer (online: the brain's curriculum, one Adam step a block; offline: epochs over 20,000 exchanges) at the brain's parameter count; a readout (one linear layer from the moment, the moment before and the question's words); a table of every situation pretraining showed; the newborn brain | the readout and the table at .50 or below on G3's panels; the brain at or above the better online control less .05 on G1 to G3 |

Every table reports parameters and multiply-accumulates (the brain: every synapse once a sweep, a
row, lessons and acts; the controls: analytic, a backward pass at twice the forward). A gate is
changed only by a declared amendment, before the run it changes.

## Files

`textworld.py` (world, grammar, parser), `textchamber.py` (the moments), `textparrot.py` (the
brain and its life), `textcontrols.py`, `textintrospect.py`, `textpistachio.py`, `textgates.py`
(lives, controls, tables), `chat.py` (talk to a trained parrot), `test_textparrot.py`.

```
python textgates.py life 33800000 d4a25 out/      # one founder's life and its readings
python textgates.py control 33800000 lstm-online out/ --brain-record out/d4a25_33800000.json
python textgates.py read out/
python chat.py out/d4a25_33800000.npz 33800000
```

## Amendment 1 (before any gated reading)

Taught on every row, a fresh brain says the majority answer ("no", .62 to .67) to "is a bottle
long ?" under every message, width and memory variant; taught only where its answer differs it
reads 1.00 by block 175 (founder 905). Every arm takes `learning_surprise=True`; the lives
started under every-row teaching were stopped before any reading. Founders and gates unchanged.

## Amendment 2 (before any gated reading)

With the value of a "what is red ?" question drawn at random, 58% of the exchanges have no red
thing, and a fresh brain stays silent at 74 of 75 matching things (block 50, founder 905). The
teacher asks about what is there four times in five (`ASKS_PRESENT`: the value of a thing in
view, or a relation some thing stands in): at block 50 the brain names 132 of 144 matching
things. Training and reading scenes alike. The lives started before were stopped before any
reading. Founders and gates unchanged.

## Amendment 3 (before any gated reading)

- **The prompt game** joins the words stage: "what is it ? W" with one thing in view, W the
  thing's name two times in three and a nonce word otherwise; the parrot says W. In development
  (founders 906, 907) a parrot taught the bare echo said "zabu" back with a lemon in view on 19 of
  20 echo trials and never said it to "what is it ?" (0 of 40 trials, "ball"); taught the prompt
  game, prompted trials carry the pistachio into the naming state (founder 906: "zabu" at 1.00 by
  trial 26).
- **Stage caps** raised to properties 600, relations 600, questions 400, scenes 600 blocks (the
  development sweep's stages ended at their caps unmastered).
- The gated lives restarted under amendments 1 to 3; no earlier life is read. G5's trials are
  declared in full before any life reaches G5.
