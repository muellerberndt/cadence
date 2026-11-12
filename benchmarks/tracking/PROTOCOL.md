# The tracking chamber's declared run (issue #320): a draft

This page is a draft. The run is declared on the issue, with the commit of this page, before the
first life of a declared founder, and it is declared once the development readings clear the
bars. A gate is changed only by a declared amendment, never after a reading. Gates S9 to S12 of
the issue (kinds and places both ways, the body's own turning and moving, a change unseen, many
things) have no panel in the chamber, and this page gains them before the declaration.

## Founders and scenes

Twelve fresh founders, 3200 to 3211; development used founders 0, 1 and 2. A founder's number
seeds its brain and every scene it trains on and is read on (`tracking.seed`).

- **Training.** `space` arms: the founder's `curriculum` of 3,000 scenes, shuffled: 25% the
  arm's own hand and 35% a person's hand in the chamber's geometry, 5% and 20% of them among
  occluding discs and dark spans of the scene's own, 10% a thing heard moving and 5% two hands
  that cross. Controls on one case: the founder's 2,000 training scenes of that case.
- **Reading.** `panels(founder)`: `own` (200 scenes), `other` (200 scenes as they come and 40 of
  each panel `straight`, `slow`, `stop`, `turn`), `varied` (100 scenes of a person's hand among
  discs and dark spans of the scene's own), `stays_own` and `stays_other` (8 scenes each, 1,000
  steps at rest out of view), `heard` (60 scenes) and `crossing` (100 scenes). Every answer is
  given by a frozen copy; nothing is taught while a panel is read.

## Arms

| Arm | What it is |
| --- | --- |
| `space` | `cadence.space` at its settings: lessons that hold only reports, the curriculum above, no map of where things are seen. Its `shift` is read at its mean. |
| `space`, read at the peak | The same lives, `shift` read at its highest peak (`read="peak"`). |
| `space`, the seen movement | The same lives, a thing that moves by itself moved by its last seen movement as it is (`read="seen"`): the control the brain's answer is read against. |
| `space-site` | As `space`, taught on the chamber's geometry alone (`vary` off) with the map of where things are seen (`sight=0.9`), read at the mean and at the peak. |
| `space-report` | As `space`, a moving thing in view taking the report itself (`smooth=False`). |
| `space-joints` | As `space`, given the arm's joint readings. |
| `last_seen`, `kalman` | The controls that learn nothing, on every panel. The Kalman filter's two noise terms are fitted on 200 training scenes a case; on `heard` it takes a cue as a report. |
| `reckoning`, `encoders` | The arm's true kinematics, on `own`. |
| `lstm-offline`, `lstm-online`, `transformer-offline`, `transformer-online` | At the `space` brain's parameter count, on `own` and on `other`, with `camera` lessons and with `truth` lessons. |

## Gates

Errors are distances in the plane, a mean over a founder's scenes and then over the twelve
founders. A standard error is over the twelve founders, of the founder-paired difference.

- **S1, in view.** `space`'s error in view on `own` and on `other` is at or under 1.1 times the
  camera's own error on the same steps.
- **S2, out of view.** On `other`: `space`'s error on the `straight` panel is at or under 1.1
  times the Kalman filter's, and on each of `slow`, `stop` and `turn` it is under the Kalman
  filter's by two standard errors. Read for `space` at the mean; the peak, the seen movement and
  `space-site` are reported beside it.
  **Proposed amendment, to be declared with the run:** on `turn` the error is at or under 1.1
  times the Kalman filter's. Its ground, on development founders 0, 1 and 2: a turn that begins
  out of view leaves no trace in what the camera reports, and the matched LSTM and transformer
  taught the true position at every step read 11.2 cm and 10.9 cm there against the Kalman
  filter's 9.0 cm.
- **S3, it stays.** On `stays_own` and `stays_other`: the answer moves by 0 over the 1,000 steps
  at rest out of view, and on `stays_other` the error over the last 100 steps is at or under the
  error while the thing was seen.
- **S4, another sense.** On `heard`: the error after the last cue is at or under the cues' own
  error (the distance from the point a cue names to the true position).
- **S5, uncertainty.** On `own` and on `other`, in view and out of view: the area at .90 holds
  the true position on .85 to .95 of the steps.
- **S6, several things.** On `crossing`: the last answers name each hand's own position in .95
  of the scenes or more.
- **S7, cost.** Milliseconds a step on one core, stated beside a camera's 33 ms, natively and
  under Pyodide.
- **S8, beside the learners.** On each of the four `other` panels, `space`'s error is at or
  under the matched LSTM's and the matched transformer's, offline, with `camera` lessons; the
  `truth` rows and the online rows are reported. The matched learners are built for scenes of
  one thing, so S6 is read beside the Kalman filter alone.

Every table states parameters, analytic multiply-accumulates of the training and milliseconds a
step. A gate that fails is reported with the term that fails it.
