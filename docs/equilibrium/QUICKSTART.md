# Build a brain, teach it a routine, keep it

Use [one continuing equilibrium brain](../world-model.md) as the application
frame: bootstrap, use, witnessed failure, local repair and continued use of the
same acquired model. This tutorial is a small calibration control for the
separate `cadence.experimental.equilibrium` population solver. Its joint
stationarity rule and explicit `History` are not the default `Brain.compose`
neural/memory implementation. Free scalar accuracy checks only this relation.

Install **Cadence 0.81.0** with Python 3.11 or later. The containing package
requires NumPy; this solver uses the Python standard library unless
you select an optional tensor backend:

```sh
python -m pip install "cadence-net==0.81.0"
```

Or install from a checkout of this version:

```sh
python -m pip install -e .
```

Retain the installed package and source when saving a reproducible experiment.
Saved brains bind their implementation sources; see [checkpoints](REFERENCE.md#checkpoints).

## 1. Build the smallest brain and teach it

We will learn a simple calibration: a supplied toy instrument produces
`movement = 0.6 * command`. Training supplies measured pairs. Later queries
supply only the command, so a correct answer must come from learned relations.
All values use the same normalized units.

A `Cortex` describes the wiring; `build()` creates the persistent `Brain`. The
smallest brain has two populations: `features` reads the sensor, `response`
reads the live states of `features`. Their patches settle against each other,
and that shared equilibrium is the answer. A single population reading only
sensors does not build: its patches would never settle with anything.

```python
from cadence.experimental.equilibrium import Brain, Cortex, bootstrap

layout = Cortex(seed=2)
command = layout.input("command", shape=1)
features = layout.column("features", patches=4, inputs=command)
response = layout.column("response", patches=1, inputs=features)
layout.output("movement", shape=1, reads=response)
brain = layout.build()

examples = [
    ({"command": [x]}, {"movement": [0.6 * x]})
    for x in (-0.8, -0.4, 0.4, 0.8)
]
checks = [
    ({"command": [x]}, {"movement": [0.6 * x]}) for x in (-0.6, 0.6)
]
report = bootstrap(
    brain, examples, checks=checks,
    epochs=20, batch_size=4, max_error=0.1, seed=2,
)
assert report["passed"], report
assert report["updates"] > 0
```

`shape` is the shape of supplied data. `patches` counts processing states.
An output exposes selected patch states; it is not a separate readout network.

`bootstrap` teaches through the normal learning calls and checks answers with
the targets absent. Its error limit measures task accuracy. The solver's
`tolerance` instead measures numerical settlement. A settled brain can answer
incorrectly; a clamped teaching output is not evidence of learning.

## 2. Test answers the teacher has not supplied

The readiness checks influence when training stops. Use different values for
this final check, with no output targets in the query:

```python
saved = brain.snapshot()
for x in (-0.7, -0.2, 0.2, 0.7):
    result = brain.settle({"command": [x]})
    assert result["qualified"], result["reason"]
    assert abs(result["outputs"]["movement"][0] - 0.6 * x) < 0.1
assert brain.snapshot() == saved
```

`settle` returns outputs and diagnostics, including `qualified`, `reason` and
`work`. Use an answer only if it qualified. `predict` is a convenient pure query
that returns just outputs and raises `SettlementError` if qualification fails.
Neither method changes the brain. This example verifies a small learned
relation, not an acquired movement policy or a general intelligence result.

## 3. Save it and use it live

```python
resumed = Brain.from_snapshot(saved)
assert resumed.snapshot() == saved
assert resumed.predict({"command": [0.5]}) == brain.predict({"command": [0.5]})

activity = resumed.step({"command": [0.3]})
assert activity["accepted"]
assert abs(activity["outputs"]["movement"][0] - 0.18) < 0.1

# After the instrument actually executes that command, its measured outcome
# can teach the same brain. Learning is still available during live use.
update = resumed.observe({"command": [0.3]}, {"movement": [0.18]})
assert update["accepted"]
```

Save the JSON text returned by `snapshot()`. It binds the layout, configuration,
parameters, activity and exact implementation sources. Keep the compatible
package, units, preprocessing and external body/history state with it.

The four main operations have different responsibilities:

| Operation | What it retains on success |
| --- | --- |
| `settle` / `predict` | Nothing; query only |
| `step` | Qualified activity; parameters stay fixed |
| `observe` | Activity and parameters taught by the supplied output witnesses |
| `observe_batch` | Shared parameters from a batch; live activity stays fixed |

A refused call does not admit its proposal. An accepted update can still worsen
other skills, so recheck free behavior after learning.

<a id="4-add-depth-when-the-task-needs-it"></a>

## 4. Add an intermediate population

For a more complicated relation, connect more populations. They all settle
together: the later relation influences earlier states through the common
energy. The body still uses the same named inputs and outputs.

```python
cortex = Cortex(seed=2)
command = cortex.input("command", shape=1)
representation = cortex.column("representation", patches=4, inputs=command)
integration = cortex.column("integration", patches=2, inputs=representation)
response = cortex.column("response", patches=1, inputs=integration)
cortex.output("movement", shape=1, reads=response)
candidate = cortex.build()

learned = bootstrap(
    candidate, examples, checks=checks,
    epochs=20, batch_size=4, max_error=0.1, seed=2,
)
assert learned["passed"], learned
assert abs(candidate.predict({"command": [0.5]})["movement"][0] - 0.3) < 0.1
```

Both layouts learn through the same patch rule. This tiny calibration already
works with the two-population brain; keep the smaller brain for this task. Add
an intermediate population when its changed paths improve measured quality
enough to justify the cost. See [layout variants](VARIANTS.md) and
[brain design](BRAIN_DESIGN.md) for larger tasks and meaningful controls.

## Experimental error readback

Every patch already repairs its own disagreement. Explicit observer wiring,
which lets a population read other patches' current errors, is available in the
[experimental layout recipes](VARIANTS.md#experimental-recursive-observer-settlement).
Those contacts participate in the same joint repair. See
[experimental capabilities](EXPERIMENTAL.md) for their behavioral scope.

## Continue with a real task

Use [bootstrapping](BOOTSTRAP.md) for data, batch learning, readiness and retention
checks. Use [live learning](LIVE.md) when actual actions have delayed outcomes or
rewards, or when a renderer needs a serial background worker. `Reinforcement`
provides explicit discrete Q-learning; a reward is not a desired motor output.
The [reference](REFERENCE.md) defines retry identities, result fields and exact
failure behavior. The repository's complete layout example also runs from a
shell:

```sh
python examples/equilibrium/layout_learning.py
python examples/equilibrium/layout_learning.py --layout deep
```
