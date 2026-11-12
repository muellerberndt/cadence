# Tracking chamber: where is the hand at this step, in view and out of view?

A plane holds an arm, a hand, a few things and a camera with a blind spot. At every step a learner
is asked one question per thing, "where is it at this step?", and its answer is scored against the
true position as a distance in the plane. The camera reports a thing only while it sees it, so the
answers that matter most are the ones given while the thing is out of view.

- [`world.py`](world.py): the world in plain NumPy. `Geometry` holds where everything stands,
  `World` what moves, and `step` advances the world by one step.
- [`tracking.py`](tracking.py): what a learner is given, the controls that learn nothing, the
  floors (a default brain asked the position at every step), the scores and the command line.
- [`controls.py`](controls.py): the matched LSTM and transformer.
- [`belief.py`](belief.py): a belief over places as a control, a grid filter with a sheet of
  mass that moves and a sheet at rest, given where the camera would see a thing at each step.
- [`scenes/promise.json`](scenes/promise.json): a recorded scene, a hand that goes behind a
  pillar at 0.5 m/s, rests there for 70 moments and comes back.
- [`scenes/example.json`](scenes/example.json): one small scene file.

```sh
PYTHONPATH=src python benchmarks/tracking/tracking.py make --case other --kind stop --seed 8 \
    --steps 60 --out scene.json                       # write a generated scene to a file
PYTHONPATH=src python benchmarks/tracking/tracking.py score --scene scene.json --arm kalman
PYTHONPATH=src python benchmarks/tracking/tracking.py life --arm code --case other --founder 0 \
    --out results/                                    # one arm, one case, one founder
PYTHONPATH=src python benchmarks/tracking/tracking.py table results/
```

## The world

Everything lies in one plane, in metres and radians, one step every 1/30 second.

- **The arm** is a chain of two links (0.45 m and 0.35 m; `THREE_JOINTS` has three) from a fixed
  base. A step's command is the joint increment asked for. The increment made differs from it by
  3% noise in proportion to its size, so an arm at rest stays where it is.
- **The camera** stands at a corner of the plane and sees a field of 120 degrees. A thing is in
  view where it lies in that field and the straight line from the camera to it crosses no
  occluding disc and no link of the arm. The disc's shadow, the arm's own shadow and the plane
  outside the field are the blind spot. In view, the camera reports a thing's position with 1 cm
  of Gaussian noise on each axis, and what the thing looks like. Out of view it reports nothing.
- **The ear** hears a thing that sounds, in view or out of it: the direction to the thing (0.04
  rad of noise) and its distance (8% of noise).
- **Things** are points. Thing 0 is the hand. Two things of one `what` look alike, and a report
  carries no name: a learner is given the list of starting positions at step 0 and has to keep
  each thing's identity itself.

Scenes, 150 steps unless a kind says otherwise:

| Case | Kind | What happens |
| --- | --- | --- |
| `own` | `sweep` | The hand is the arm's own end. The arm moves between joint targets over 25 to 60 steps with rests of up to 25 steps, from a list of starting poses. |
| `own` | `stays` | One move into the blind spot, then 1,000 steps at rest there. |
| `other` | `straight`, `slow`, `stop`, `turn` | The hand is a person's and the arm holds a pose. The hand leaves a starting position toward the plane's middle at 0.25 to 0.6 m/s. At a step drawn between 15% and 70% of the scene it eases to 25% to 50% of its speed, eases to rest, or turns by 40 to 110 degrees; a `straight` hand keeps its line. |
| `other` | `stays` | A thing rests in view for 30 steps, the arm moves until its shadow holds the thing, and the thing rests unseen for 1,000 steps. |
| `room` | `heard` | A thing rests in the blind spot, then moves to another point of it without coming into view, sounding while it moves and for 3 steps after. Two further things rest in view and sound at times. |
| `room` | `crossing` | Two hands that look alike cross inside the blind spot, both out of view, among six things at rest. |
| `room` | `demo` | A person's hand that reaches and comes back, two things at rest, and one that is heard moving out of view. |

A change of movement falls where the draw puts it, in view or out of it, so a learner that only
reads the camera sees every kind of movement. The panels `straight`, `slow`, `stop` and `turn`
hold the scenes whose change begins out of view and stays there for its 8 steps (for `straight`:
8 or more steps in a row out of view).

## A scene file

One file per scene: JSON, or a NumPy `.npz` with the same names (its `meta` entry holds the JSON
of every field that is no per-step array, and NaN stands where the JSON holds `null`). Metres,
seconds and radians.

```text
{
 "format": "cadence-tracking-scene/1",
 "units": {"length": "m", "time": "s", "angle": "rad"},
 "dt": 0.0333,                    seconds a step
 "case": "own" or "other",        the arm's own end, or a person's hand
 "geometry": {
   "plane": [x_low, x_high, y_low, y_high],
   "camera": {"position": [x, y], "heading": rad, "fov": rad, "range": m},
   "occluders": [{"center": [x, y], "radius": m}],
   "arm": {"base": [x, y], "lengths": [m, m], "width": m, "limits": [[low, high], [low, high]]}
 },
 "start": [x, y],                 the hand's starting position, from the list
 "report":   one row a step: [x, y] while the hand is in view, null while it is not
 "in_view":  one row a step: true or false
 "commands": one row a step: the joint increments asked for between the step before and this one
 "joints":   one row a step: the joint readings at this step
 "truth":    one row a step: [x, y], the true hand position at this step
}
```

- Joint 0's angle is measured from the x axis and every further joint's from the link before it,
  counter-clockwise. `commands` holds zeros at step 0.
- A file holds `format`, `case`, `report`, `in_view` and `truth` at the least. Where it leaves
  them out, `start` is the truth at step 0, `dt` is 1/30, `commands` and `joints` are an arm at
  rest and `geometry` is the chamber's own.
- Optional: `camera_noise` (the report's standard deviation on each axis), `kind`, `event` (the
  step a movement changes), `name`.
- A scene of several things holds `report` and `truth` as `(steps, things, 2)`, `in_view` as
  `(steps, things)`, `start` as `(things, 2)`, `what` (one number a thing: what it looks like) and
  `cue` (per step and thing, `[direction, distance]` from `geometry.ear` while it sounds, else
  `null`).
- The scores read `report`, `in_view`, `commands`, `joints` and `truth`. The geometry draws the
  scene and lets the chamber generate training scenes like it.

`world.load(path)` reads a file and `world.save(scene, path)` writes one. The chamber scores a
loaded scene with the calls it uses for a generated one, and
[`test_tracking.py`](test_tracking.py) holds that for the example file.

## What a learner is given, and the lessons

At every step: the reports of the things in view (position and look), the cues, the arm's
commands, and at step 0 the list of starting positions. The joint readings are in every scene and
are given only where an arm says so (`--joints`): with them the arm's own hand is a fixed function
of the readings.

Two kinds of lessons:

- `truth`: a training scene teaches the true position at every step, in view and out of view.
- `camera`: a lesson holds only positions the camera gave. A learner may hide reports from
  itself and be taught by the report it did not read.

## Arms

| Arm | What it is |
| --- | --- |
| `last_seen` | Each thing where it was last reported. |
| `kalman` | A constant-velocity Kalman filter per thing. Its process noise and report noise are fitted on the training scenes, at the least error out of view. Reports of things that look alike go to the nearer places. |
| `reckoning` | The arm's true kinematics on the commands, from the last report. It knows the arm's lengths, base and bend, which no learner is given. |
| `encoders` | The true kinematics on the joint readings. |
| `default` | A default brain on the readings as they come, asked the position at every step: two motor slots of 32 tuned positions, one an axis. |
| `code` | A default brain on population-coded readings (`PopulationCode`), a report out of view and a command of zero dark. |
| `lstm-offline`, `lstm-online`, `transformer-offline`, `transformer-online` | An LSTM and a causal transformer of two layers at the brain's parameter count, trained by gradient on the same scenes. Online: one pass in the brain's order. Offline: epochs until the loss on scenes held aside stops falling. |

## Scores

Distances in the plane between an answer and the true position, over held-out scenes:

- the hand in view and out of view, and by the steps since it was last seen (1 to 5, 6 to 10, 11 to
  20, 21 to 40, 41 to 80, 81 and more);
- per panel, over the panel's steps out of view;
- `stays`: the error over the last 100 of 1,000 steps out of view, beside the error while seen;
- `heard`: the error after the last cue, beside the cues' own error;
- `crossing`: the share of scenes whose last answers name each hand's own position;
- where an arm states an area at .90, the share of steps whose area holds the true position.

Every table states parameters, analytic multiply-accumulates of the training and milliseconds per
answer on one core.

## The map of things

[`mapped.py`](mapped.py) scores a scene with `cadence.space.map`, the map of things held in a
sheet's own settled activity ([the guide](../../docs/space.md)). The map is given what the camera
gives: a report of the hand while it is seen and, at every step, the cells the camera sees. Its
answer is the mean place of the hand's activity. Last seen and the Kalman filter are printed
beside it, with the error in view, out of view and by the steps since the hand was seen:

    python benchmarks/tracking/mapped.py --scenes benchmarks/tracking/scenes/promise.json
    python benchmarks/tracking/mapped.py --founder 0 --reach 0.15 0.3 0.6 --empty near keep

`--empty` and `--reach` name the rule for a thing whose place is seen empty. With `keep` the map
answers as last seen does, from its own carried activity.

`--physics '{"per": 3}' --train 800 --watched 0 50 200` raises learned physics on a nursery of
the founder's scenes (`nursery`, the shares in `MIX`) and reads it after each number of scenes
watched. [`belief.py`](belief.py) is a classical grid filter, a control.
