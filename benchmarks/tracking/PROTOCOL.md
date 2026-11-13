# The map of things: the declared run of its core gates (issue #320)

This page is declared on the issue, with the commit that holds it, before the first life of a
declared founder. A gate is changed only by a declared amendment, never after a reading.

## What is read

The map of things, `cadence.space.map.Map`, at its founder settings: 32 by 26 cells of the
chamber's plane, 16 code units (64 for 128 kinds, three units a kind), `empty="near"`,
`reach=0.15`, `step=0.15`, `stride=0.5`, `margin=0.02`, no physics. The map has no taught
synapse: a founder seeds the scenes it is read on and nothing else. Learned physics on the map
is read apart (the second stage) and is no part of this run.

## Founders and scenes

Twelve fresh founders, 3200 to 3211; development used founders 0, 1 and 2. A founder's number
seeds every scene (`tracking.seed`). Each founder runs two commands at the declared commit:

    python benchmarks/tracking/mapped.py --founder F --test 200 --quota 40 --each 12 \
        --own 200 --per --stays 8 --heard 60 --out mapped-F.json
    python benchmarks/tracking/rooms.py --founder F --rooms 20 --out rooms-F.json

- **A person's hand** (`other`): 200 scenes as they come and 40 of each panel `straight`, `slow`,
  `stop`, `turn`. **Errands**: 12 scenes each of hands left out of view, that visit a shadow and
  come back, pause, hop and go slowly. The camera's sight is given as a function of places.
- **The arm's own hand** (`own`): 200 scenes; the map is given the place the arm's joint readings
  name by the program's forward kinematics while the hand is out of view.
- **At rest** (`stays`): 8 scenes of a thing at rest out of view for 1,000 moments.
- **Heard** (`heard`): 60 rooms in which a thing at rest out of view is heard moving to another
  hidden place; each cue's place (the ear's direction and distance) is a report of `heard=`.
- **Rooms** (`rooms.py`): a camera whose field of 40 degrees turns over the room once and then
  over one half. `kinds` (eight things of four kinds), `moved` (a thing moved while the camera
  looks away), `many` (8, 32 and 128 things of as many kinds), `arrive` (things arrive, one is
  taken away unseen and put back elsewhere), and the state check.

Controls on the hand panels: last seen and the constant-velocity Kalman filter (its two noise
terms fitted on 100 training scenes of the founder). The map at `empty="keep"` is read beside it.

## Gates

Means over a founder's scenes, then over the twelve founders; a standard error is over the
founders.

- **S1, in view.** On `other` and on `own`, the map's error in view is at or under 1.1 times the
  camera's own error on the same steps.
- **S3, it stays.** On `stays`, the answer moves by 0 over the hidden moments, and the error over
  the last 100 of them is at or under the error while the thing was seen.
- **S4, another sense.** On `heard`, the error after the last cue is at or under the cues' own
  error (the distance of the last three cues' points from the true place).
- **S7, cost.** Milliseconds a moment natively on one core and under Pyodide, stated beside a
  camera's 33 ms; the page's bar is 30 ms under Pyodide with four things.
- **S9, what is where.** In `kinds`, for the things in view and for those last seen 500 moments
  before: "where is" each kind holds each of its things within 5 cm at .95, the mean distance of
  the nearest place is at or under 1.5 times the seen error (the camera's mean error, 1.25 cm),
  and "what is at" each thing's place names its kind at .95. Two things of one kind are two
  places.
- **S10, the turning camera.** The things last seen 500 moments before in `kinds` are those the
  turning camera left behind: they are read as S9 says. The body that moves is not built and is
  reported open.
- **S11, a change unseen (as amended).** In `moved`, the old place is held until the camera sees
  there again (.95), and within five moments of the camera having seen both the new place and
  the old place the new place is reported and nothing is at the old (.95).
- **S12, many things.** In `many`, 8, 32 and 128 things are each held by kind (where within
  5 cm) and by place (what) at .95.
- **S13, what there is.** In `arrive`, the inventory lists the things present within 5 cm, with
  no extra and none missed, at .95 of the things at every reading; the thing taken away is "not
  known where" after the camera has swept over its place; put back elsewhere it is found as one
  thing.
- **S14, the state.** On `stays` the hidden answer moves by 0 for 1,000 moments; in the state
  check the map after 1,000 moments with nothing seen holds its eight things within 5 cm, a
  file written and read back answers alike, and the emptied map reports no thing.

Reported beside the gates, no gate: the hand's error out of view by panel and by steps unseen,
the area (the smallest set of cells holding .90 and .95 of the hand's activity) and its coverage,
beside last seen and the Kalman filter; the two recorded scenes `scenes/promise.json` and
`scenes/promise-folded.json`, which took no part in any choice.

Dropped from this run, with the reason: S2, S5, S6 and S8 are gates of the second stage (learned
physics) and of anticipation; the map does not move a thing it does not see. S10's moving body is
not built.

# Learned physics on the map: the declared run of its gates (issue #324)

Declared on the issue, with the commit that holds it, before the first life of a declared
founder. A gate is changed only by a declared amendment, never after a reading.

## What is read

The map of things at its founder settings (as above) with learned physics at its founder
settings, `Map(physics={})` (`cadence.space.physics.Physics()`: `noise` 0.01, `line` 30,
`surprise` 3.0, `settled` 1.5, `every` 8, `horizon` 30, `reports` 10, `prior` 500). The control is
the same map without physics, which the physics before any lesson equals moment for moment.

## Founders, the world watched and the scenes read

Twelve fresh founders, 3300 to 3311; development used founders 0, 1 and 2. A founder's number
seeds every scene. Each founder runs one command at the declared commit:

    python benchmarks/tracking/mapped.py --founder F --test 200 --quota 40 --each 12 \
        --scenes benchmarks/tracking/scenes/promise.json benchmarks/tracking/scenes/promise-folded.json \
        --physics '{}' --watch 400 --watched 0 25 50 100 200 400 --out physics-F.json \
        --save taught-F.npz

- **Watched** (`reaches`): 400 scenes of a person's hand drawn as the test scenes are (straight,
  slow, stop, turn), from the founder's training seed, half in the chamber's geometry and half
  among discs of the scene's own (with the camera off for a span in one of three). The map
  watches them in order; its physics learns while it watches.
- **Read** after 0, 25, 50, 100, 200 and 400 scenes watched, with the physics' learning paused:
  a person's hand (200 scenes as they come and 40 of each panel `straight`, `slow`, `stop`,
  `turn`) and the errands (12 scenes each of hands left out of view, that visit a shadow and come
  back, pause, hop and go slowly). Controls: last seen, the Kalman filter (its two noise terms
  fitted on 100 training scenes of the founder) and the map without physics.

## Gates

Means over a founder's scenes, then over the twelve founders; a standard error is over the
founders. "Out of view" is the hand's error over every step it is out of view.

- **P1, out of view.** On a person's hand after 400 scenes watched, the map with physics is under
  the map without it by two standard errors of the founders' differences.
- **P2, straight.** On the `straight` panel after 400 scenes watched, the error is at or under
  2 times the Kalman filter's.
- **P3, stops.** On the `stop` panel after 400 scenes watched, the error is at or under the map's.
- **P4, the area.** After 400 scenes watched, the smallest set of cells holding .90 of the hand's
  activity holds the true place on .90 of the steps out of view.
- **P5, the curve.** At every number of scenes watched (0, 25, 50, 100, 200, 400), the error out
  of view is at or under the map's.
- **P6, cost.** With two hands moving out of view (`moving.py --map taught-3300.npz`), a moment
  and its inventory take under 20 ms at the median under Pyodide 314.0.7 on a machine running
  nothing else; the native time is stated beside it.

Reported beside the gates, no gate: every panel, the errands and the steps since last seen, the
area at .95, the four numbers learned and the lessons behind them, the two recorded scenes
`scenes/promise.json` and `scenes/promise-folded.json`.
