# #332, protocol 1: the parrot's first brain on twelve fresh founders

Declared before any gated life. The code is `benchmarks/parrot/` and `src/cadence/parrot/` at
the commit this file is committed with; the selected genome is frozen in
`benchmarks/parrot/selected.json` before the first gated life.

## The world and the life

- **The table** (1.6 by 1.1 m) and a low perch. Each scene holds 3 to 6 things of eight kinds,
  0.22 to 0.6 m from the perch, all the way around it, 0.18 m apart or more, each on a clear
  stand 0 to 0.25 m high at a scale of 0.7 to 1.4 and a random turn. A stand and a scale make a
  thing's height in the field and its size in the eye tell little about how far it is, so the
  two eyes and the head's own movement are what tell it. The panel of things standing on the
  table is read beside it, with no gate.
- **The eyes**: 175 by 100 pixels each at one degree a pixel; the fields overlap 25 degrees in
  front of the beak and leave 35 degrees unseen behind. The neck turns the head 75 degrees either
  way, so a thing more than 87.5 degrees from the beak's rest is never seen by both eyes.
- **A life**: 48 scenes of 50 moments (2,400 moments, four minutes of the parrot's time). The
  parrot looks where its focus takes it. Its only lessons of distance are its own: two sightings
  of one thing from two places of the eyes (the head turned in between) meet at one place, and
  the distance from each eye to that place is the lesson for what that eye saw. No distance,
  place or kind is given to the brain.

## Founders and arms

Founders 33200000 to 33200011 (twelve, fresh). Development founders 900 to 905 and the selection
worlds (33289000 to 33299999) take no part in a gated reading.

| Arm | What it is |
| --- | --- |
| Selected | The lineage's best genome: `benchmarks/parrot/evolve.py`, seed 33290000, 16 generations of 32, 2 worlds each, keep 4, mutation rate 0.15 over 25 named genes |
| Hand-set | Every gene at the founder's value (`cadence.parrot.brain.GENES`) |
| Random wiring | The hand-set genes; the reach's synapses as many as the hand-set's, their ends drawn at random among its inputs and distance neurons |
| Random search | The best of as many uniform draws over the same 25 genes as the lineage scored, on the same worlds |
| CNN, ViT (E5) | In place of the distance neurons: a convolutional network and a vision transformer that see both eyes' whole images and where the sighting is in them, taught by the same lessons as they come, at the selected reach's parameter count; and at ten times it, read with no gate |

Every table carries learned parameters and multiply-accumulates a moment.

## Readings (on fresh test scenes after the life, no lessons)

- **E1, seen and held.** After a 60-moment look-around by the parrot's own focus, the share of
  things the map holds within 0.10 m of their place. Then each thing so held that the neck can
  put in the blind spot is put there for 20 moments: the share held within 0.10 m after them.
- **E2, one thing from another view.** Each thing seen by the left eye alone, the right eye
  alone and both, as far as the neck reaches (two views or more), then 30 moments of the
  parrot's own looking: the share held as exactly one thing within 0.15 m, that one within
  0.10 m.
- **E3, distance learned.** The median |log(read / true)| of the distance read for each thing
  the neck brings in front of the beak, by both eyes, and the same things with the left eye or
  the right eye covered. The learning curve: the same error over the sightings by both eyes in
  the life's first and last quarter.
- **E7, the distance neurons.** A probe panel: one thing at a time in front of the beak at 8
  distances from 0.22 to 0.6 m, at a new scale and stand each time, 24 things. Every neuron of
  the cue, distance, look and kind regions gets its information about distance (bits, less the mean with the
  distances shuffled) and its tuning curves to distance and to disparity. The found neurons:
  information at half the most informative neuron's or more. The tests are read again with the
  found neurons held at rest inside every settle, and with as many neurons drawn at random from
  the brain's other settling neurons (the cue, distance, look, kind and focus regions outside
  the found ones) held at rest.
- **E4, the wiring selected.** The fitness of each arm on each founder's own worlds (the terms
  of `evolve.py`, each over the hand-set's on the same founder).
- **E6, in the browser.** The page's host renders both eyes and lives one moment in Pyodide in
  headless Chrome; the median time a moment over 600 moments, by part.

## Gates

A threshold gate holds when the mean over the twelve founders reaches it and at least 10 of 12
founders do.

| Gate | Arm | Threshold |
| --- | --- | --- |
| E1 seen | Selected | .85 within 0.10 m |
| E1 behind | Selected | .95 held after 20 moments in the blind spot |
| E2 | Selected | .85 |
| E3 learned | Selected | both eyes in front, median error .08 or less; the last quarter of the life below the first |
| E3 one eye | Selected | covered over both eyes 1.25 or more on the mean, above 1 on 10 of 12 |
| E7 found | Selected | the found neurons hold 80% or more of the distance neurons; tuning curves drawn |
| E7 silenced | Selected | E3 error with the found neurons silenced 2 times the intact or more; bearing of held things within 1 degree of the intact; E2 within .10 of the intact |
| E7 others | Selected | E3 error with as many other neurons silenced at most 1.25 times the intact plus .01 |
| E4 | Selected against hand-set | higher fitness on the mean and on 8 of 12 founders |
| E4 | Hand-set against random wiring | higher fitness on the mean and on 10 of 12 founders |
| E4 | Selected against random search | higher fitness on the mean |
| E5 | Selected against CNN and ViT | lower E3 error (both eyes, front) on the mean and on 10 of 12 founders, at matched parameters |
| E6 | Selected | 100 ms a moment or less (ten moments a second), the cost stated |

## Expectations, from development founders 900 to 905 (hand-set, 24 scenes; no gated life)

- E1 within 0.10 m: .68 to 1.0 (mean .82). Things both eyes saw: .75 to 1.0 within 0.08 m;
  things one eye saw: .54 to .73.
- E3 both eyes in front: .047 to .109; one eye covered over both eyes: 1.07 to 2.04 (mean
  1.58).
- One moment on one core of the box: 23 ms (render 9, early vision 6, map 4).
