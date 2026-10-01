<p align="center">
  <img src="https://raw.githubusercontent.com/muellerberndt/cadence/main/docs/assets/cadence-logo.png" alt="Cadence: learning through flat, deep and recursive settlement" width="100%">
</p>

# Cadence

[Documentation](docs/index.md) · [Quickstart](docs/QUICKSTART.md) · [Examples](examples/README.md) · [Interactive overview](https://floatingpragma.io/cadence/) · [Preprint](https://philpapers.org/rec/MUECAP-2) · [Pragma Research](https://floatingpragma.io/)

[![PyPI](https://img.shields.io/pypi/v/cadence-net)](https://pypi.org/project/cadence-net/)
[![CI](https://github.com/muellerberndt/cadence/actions/workflows/ci.yml/badge.svg)](https://github.com/muellerberndt/cadence/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/cadence-net)](https://pypi.org/project/cadence-net/)
[![License: GPL v3](https://img.shields.io/badge/license-GPLv3-blue.svg)](LICENSE)

**Build a learned routine, add depth where it helps, and experiment with recursive feedback.**

Cadence is an experimental learning library built from bounded patches with
local state, input and output ports, retained relations and prediction-error
readback. Connected patches settle together to answer; learning repairs their
relations from experience. Recursive observers read other patches' live states
and exact errors, feeding back into that same settlement. This is the
observer-like, self-reading structure behind Cadence.

**This README describes `0.60.0.dev0` on `main`.** It supports flat, ordinary deep,
recursive and mixed layouts through one API. The intended System 1/System 2
behavior is **routine is cheap; disturbance recruits useful correction; learned
correction becomes routine**. Automatic internal attention, independently
progressing fast and slow populations, and a demonstrated recursive advantage
remain requirements for stable 0.60.0. The development version does not yet
provide that complete cycle. See [what is available](docs/MIGRATION_060.md).

## Choose how the brain responds

| Layout | What it does | When to start here |
| --- | --- | --- |
| **Flat: a simple routine response** | Each patch reads sensors directly. With parameters fixed, patch states can settle independently. | Small direct mappings, calibration and the first inexpensive baseline. |
| **Deep: co-settling representations** | Ordinary populations read earlier populations' live states. All layers settle together, with returning influence through their shared constraints. | Routines that need learned intermediate features or combined sensory information. |
| **Recursive: state-and-error feedback** | Observers read live states and exact current prediction errors. An observer can itself be observed. | Testing whether internal error readback improves correction beyond capable ordinary layers. |

All three use the **same patch rule**, learning methods and whole-brain
qualification. These are choices of wiring, not three neuron classes or speed
settings. A flat layout is a useful fast baseline; actual latency depends on
size, coupling, learning and the task. Recursive depth alone does not make a
brain more capable, and it has no separate slow clock in the current runtime.

**System 1** means learned routine competence; it can need several ordinary
layers. **System 2** means extra recursive correction when routine competence
fails. A coherent changing beat, a familiar game situation or walking a known
path can all be routine. Equilibrium in this behavioral sense means sustained
competence, not an unchanging output. Numerical settlement alone can still give
a wrong answer about the world: compare forecasts with later observations and
measure actual task outcomes.

Combine these layouts inside one `Cortex`. The application supplies observations,
executes actions and reports outcomes through one brain/body interface; it does
not attach an evaluator to every population. Start with the
[three layout quickstarts](docs/VARIANTS.md), including a
[mixed brain](docs/VARIANTS.md#combine-routine-layers-and-recursive-observation).

## Install the development version

Python 3.11 or later. The default engine needs only the standard library:

```sh
python -m pip install "cadence-net @ git+https://github.com/muellerberndt/cadence.git@main"
python -c "import cadence; print(cadence.__version__)"
```

For reproducible work, replace `main` with the reviewed commit hash. The earlier
published baseline is `cadence-net==0.50.0`; its API differs from this guide.
See [migration and checkpoint compatibility](docs/MIGRATION_060.md).

Optional PyTorch execution uses the same learning rule and final reference
check. Install the development source with its `gpu` extra, then choose
`Cortex(device="cpu")`, `Cortex(device="mps")` or `Cortex(device="cuda")`:

```sh
python -m pip install "cadence-net[gpu] @ git+https://github.com/muellerberndt/cadence.git@main"
```

Small brains can be faster on the default engine. Measure the complete workload;
see [devices, precision and batching](docs/ACCELERATION.md).

## Teach a small body model

This flat brain learns how a supplied one-dimensional simulator moves. It sees
position and commanded velocity, then predicts the next position. Teaching,
readiness checks and final probes use different inputs. Predictions come from
the settled brain; the simulator supplies only measured teaching and test values.

```python
from cadence import Brain, Cortex, bootstrap

# Supplied simulator: one quarter-second of movement.
def advance(position, velocity):
    return position + 0.25 * velocity


def measurements(positions, velocities):
    return [
        ({"body": [x, u]}, {"next_position": [advance(x, u)]})
        for x in positions for u in velocities
    ]


layout = Cortex(seed=2)
body = layout.input("body", shape=2)
response = layout.column("response", patches=1, inputs=body)
layout.output("next_position", shape=1, reads=response)
brain = layout.build()

report = bootstrap(
    brain,
    measurements((-0.5, 0.5), (-0.8, 0.8)),
    checks=measurements((-0.25, 0.25), (-0.4, 0.4)),
    max_error=0.06, epochs=30,
)
assert report["passed"], report

# Final free predictions: no answer is clamped or supplied as an input.
for position, velocity in ((0.35, -0.4), (-0.35, 0.4)):
    forecast = brain.predict({"body": [position, velocity]})["next_position"][0]
    assert abs(forecast - advance(position, velocity)) < 0.06

# Live operation: retain activity, execute, then learn the actual consequence.
inputs = {"body": [0.35, -0.4]}
activity = brain.step(inputs)
assert activity["accepted"]
measured = advance(0.35, -0.4)
assert brain.observe(inputs, {"next_position": [measured]})["accepted"]

saved = brain.snapshot()  # JSON text: state, learned relations and source identity
restored = Brain.from_snapshot(saved)
assert restored.predict(inputs) == brain.predict(inputs)
```

This learns a small forward model, not a navigation policy. The
[live-control example](examples/live_control.py) uses a learned model to compare
candidate actions and move an actual simulated body toward a goal. Its action
search is supplied application code. It does not demonstrate automatic attention
or a benefit from recursion.

`settle` and `predict` query without changing the brain. `step` retains qualified
activity. `observe` also learns from supplied output witnesses; `observe_batch`
learns from several independent examples while preserving live activity.
Check `qualified` or `accepted`; `predict` raises `SettlementError` on refusal.
A teaching clamp matching its target is not evidence of learning—check later
predictions without targets.

## Run the current examples

From a checkout of this development version:

```sh
python -m pip install -e .
python examples/layout_learning.py --layout all
python examples/live_control.py --decisions 20 --seed 0
python examples/live_learning.py --seeds 0 2 7
```

| Example | What you can verify |
| --- | --- |
| [Three layouts, one interface](examples/layout_learning.py) | Acquire a small relation using flat, deep and recursive layouts; check fresh predictions, work counts and exact saved continuation. These have different capacities and are not an advantage comparison. |
| [Learned body control](examples/live_control.py) | Bootstrap a body model, select actions through explicit candidate search, execute them and admit actual outcomes. |
| [History, retention and rewards](examples/live_learning.py) | Separate small tests of explicit sensory history, old-skill replay, reward learning/reversal and saved continuation. |

[All examples](examples/README.md) include batch learning, independent parallel
brains, delayed reward and layout costs. `History` supplies explicit external
memory; `Reinforcement` supplies discrete action-value learning and replay.
Neither is an automatic planner or a guarantee of long-term success.

The [public demos](https://github.com/muellerberndt/cadence-demos) also include
Amen, Atari, Patch World and Doom experiments. Their original engines and
results have different versions and representations. They are **historical
application evidence**, not completed reproductions on 0.60.0. In particular,
the original Amen record-cell brain is not equivalent to one current flat
patch. Use the [demo migration guide](docs/MIGRATION_060.md#reproduce-the-website-demos-before-optimizing-them)
and [versioned performance evidence](docs/PERFORMANCE.md) before comparing them.

## Learn more

| Guide | What it helps you do |
| --- | --- |
| [Quickstart](docs/QUICKSTART.md) | Build, teach, query and save your first brain |
| [Layout quickstarts](docs/VARIANTS.md) | Construct flat, deep, recursive and mixed brains with the same interface |
| [Brain design](docs/BRAIN_DESIGN.md) | Choose sufficient observations, connected capacity and useful evaluation checks |
| [Bootstrapping](docs/BOOTSTRAP.md) | Prepare a skill and measure acquisition, retention and learning cost |
| [Live operation](docs/LIVE.md) | Connect observations, actual outcomes, history, reward and control callbacks |
| [Agent recipe](docs/AGENTS.md) | Build integrations with the right contracts and capability claims |
| [Architecture](docs/DRSN.md) / [API reference](docs/REFERENCE.md) / [Specification](docs/SPECIFICATION.md) | Understand the equations, exact calls and numerical guarantees |

Qualification means constrained numerical stationarity, not a unique global
minimum or task success. Saved brains bind exact implementation sources; retain
those sources and application preprocessing with checkpoints. Broader capability
and comparative efficiency require measured task evidence.

## Development

Changes follow **minimalism**, **user-friendliness** and **agent-friendliness**.
See the [contributor instructions](AGENTS.md).

```sh
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check src tests
python -m ruff format --check src tests
```

Tests execute the README and documentation examples and check learning,
mathematical derivatives, refusal and checkpoint continuation.
Licensed under [GPL-3.0-or-later](LICENSE).
