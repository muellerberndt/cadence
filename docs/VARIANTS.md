# Flat, deep and recursive layout quickstarts

Cadence supports three design patterns: **flat settlement**,
**deep ordinary (state-coupled) settlement**, and **recursive observer settlement**.
All three use the same patch law, repair engine, learning operations and qualification
check. Recursive observation is a wiring choice, not a prerequisite for
settlement.

These examples target **0.60.0.dev0**; follow the [quickstart installation
instructions](QUICKSTART.md) before running them. “Fast” describes the
intended cost of a learned routine, and “slow” the extra work a correction may
need. They are not selectable execution modes. All populations currently take
part in one qualified solve, including observers in a mixed layout.

| Design pattern | What patches read | When to try it |
| --- | --- | --- |
| Flat settlement | Fixed sensory inputs only | A direct sensor-to-answer mapping; start here for small, latency-sensitive tasks |
| State-coupled settlement | Sensors and other populations' live states | A task that needs learned intermediate representations or sensory fusion |
| Recursive observer settlement | Live states **and exact prediction errors**, including those of other observers | Test whether interpreting internal disagreements improves the task beyond ordinary composition |

These are layout patterns within one implementation. They can coexist in a
brain: sensory columns, state-reading populations and nested observers can
participate in one joint settlement. This does not automatically give the flat
part an independent deadline; the full query must qualify before it returns.

Choose a construction below, then use the shared learning/query/save loop.
All three expose `signal` and `answer`; the body interface does not change.
These are teaching examples with different capacities and contact counts, not
a controlled comparison of architecture quality. The complete runnable version
is [layout_learning.py](../examples/layout_learning.py):

```sh
python examples/layout_learning.py --layout flat
python examples/layout_learning.py --layout deep
python examples/layout_learning.py --layout recursive
```

## 1. Flat settlement

Each patch reads fixed sensors. With parameters held fixed for a query, the
state objective separates into independent scalar problems. This makes a
small flat layout a useful starting point for direct control, calibration or
simple prediction tasks.

```python
from cadence import Cortex

flat = Cortex(seed=7)
signal = flat.input("signal", shape=1)
response = flat.column("response", patches=1, inputs=signal)
flat.output("answer", shape=1, reads=response)
flat_brain = flat.build()

result = flat_brain.settle({"signal": [0.4]})
assert result["qualified"]
```

This is still a settlement operation. The current engine runs its ordinary
repair and final stationarity check. The [performance guide](PERFORMANCE.md)
derives the special flat query's exact optimum and explains why learning can
cost more than querying it.

## 2. Deep ordinary settlement

Pass a population through `inputs` to read its states. This ordinary composition
lets output patches use a learned intermediate representation. Populations
settle together; the downstream relation can influence upstream states through
the joint energy's derivatives, even without any error-reading observer.

```python
from cadence import Cortex

composed = Cortex(seed=7)
signal = composed.input("signal", shape=1)
representation = composed.column("representation", patches=4, inputs=signal)
integration = composed.column("integration", patches=2, inputs=representation)
response = composed.column("response", patches=1, inputs=integration)
composed.output("answer", shape=1, reads=response)
composed_brain = composed.build()

result = composed_brain.settle({"signal": [0.4]})
assert result["qualified"]
```

The read graph looks layered, but execution jointly repairs its live states.
It is not a single feed-forward pass through frozen intermediate activations.
Choose this pattern when a direct mapping is insufficient; measure whether the
learned representation improves behavior enough to justify the coupling cost.

## 3. Recursive observer settlement

Use `observes` to read a population's states and exact current prediction
errors. An observer can then observe another observer. All of these relations
participate in the same objective; higher populations do not wait for lower
ones to finish and then issue a separate correction.

```python
from cadence import Cortex

recursive = Cortex(seed=7)
signal = recursive.input("signal", shape=1)
representation = recursive.column("representation", patches=4, inputs=signal)
integration = recursive.observer(
    "integration", patches=2, observes=representation,
)
reflection = recursive.observer(
    "reflection", patches=1, observes=integration,
)
recursive.output("answer", shape=1, reads=reflection)
recursive_brain = recursive.build()

result = recursive_brain.settle({"signal": [0.4]})
assert result["qualified"]
```

This pattern supplies explicit internal state-and-error readback. Test it on
tasks where that extra information could help resolve competing constraints.
Its usefulness must be learned and measured against the state-coupled control;
more observer levels alone do not establish better reasoning. The
[DRSN guide](DRSN.md) explains the coupled equations and returning influence.

## Teach and save through the same interface

Only the construction changes. This task is `answer = 0.6 * signal`; it needs no
recursive observer. Training witnesses and development checks are distinct from
the final free query, and all answer values are absent from query inputs.

```python
from cadence import Brain, bootstrap

examples = [
    ({"signal": [x]}, {"answer": [0.6 * x]})
    for x in (-0.8, -0.4, 0.4, 0.8)
]
checks = [
    ({"signal": [x]}, {"answer": [0.6 * x]}) for x in (-0.6, 0.6)
]
for brain in (flat_brain, composed_brain, recursive_brain):
    report = bootstrap(
        brain, examples, checks=checks, epochs=20,
        max_error=0.1, batch_size=4, seed=2,
    )
    assert report["passed"], report
    saved = brain.snapshot()
    result = brain.settle({"signal": [0.5]})
    assert result["qualified"]
    assert abs(result["outputs"]["answer"][0] - 0.3) < 0.1
    restored = Brain.from_snapshot(saved)
    assert restored.settle({"signal": [0.5]}) == result
    assert brain.snapshot() == saved
```

## One execution and learning contract

For every pattern, `settle` and `predict` query without changing continuation;
`step` retains qualified activity; `observe` learns from supplied witnesses;
and `observe_batch` learns shared parameters from private example states.
Check `qualified` or `accepted` before using results. `predict` raises on
refusal. The [API reference](REFERENCE.md) describes their exact contracts.

Start with the least expensive pattern that learns the behavior. Measure task
quality, query latency, learning cost and refusals separately. Coupling can add
substantial work, but recursive layouts do not have a universal speed ordering
relative to ordinary composition. See the [performance guide](PERFORMANCE.md)
and [runnable examples](../examples/README.md) for measured comparisons.

## Choose routine competence separately from observation

A familiar skill can require a deep learned representation. “System 1” does
not mean one flat layer, and “System 2” does not mean any population named
`reflection`. Ordinary composition already has returning influence during joint
repair. An observer adds the exact **current** error signal, not an independent
critic, a recorded past failure or a built-in long-term objective.

The public builder reads previously declared sources only. Its read graph is
acyclic even though solving the common energy returns influence upstream.
`step` preserves activity; that alone does not demonstrate learned recurrent
memory. See [brain design](BRAIN_DESIGN.md#measure-speed-and-retained-correction)
for the current attention boundary and temporal controls.

## Width, branches and readouts

| Choice | Construction | Meaning |
| --- | --- | --- |
| Width | `column(patches=256, inputs=...)` | 256 processing states and their incoming relations |
| Parallel branches | Multiple columns reading different sensors | Separate sensory representations participating in the same solve |
| Ordinary composition | `column(inputs=(vision, hearing), ...)` | A population reading other populations' states |
| Recursive observation | `observer(observes=(vision, hearing), ...)` | State **and exact live error** readback with feedback in the joint energy |
| Deeper observation | An observer includes an earlier observer in `observes` | Observation of a system that already observes other patches |
| Motor readout | `output(shape=(8,), reads=population)` | Eight selected patch states, with no separate readout network |

An observer can receive ordinary `inputs` too. Extra depth gives additional
coupled constraints and capacity; it does not grant a final veto or guarantee
better reasoning. A population's role is its wiring, not a different neuron
class. Sensory `shape` describes supplied data, not learned interpretation.

Patches in a flat population have separate incoming relations. Selecting one
as an output does not give it access to all the other patches: connect a second
population to the first to create a learned hidden representation. More unused
flat patches are not a substitute for that connection. See
[bootstrapping](BOOTSTRAP.md) for small starting sizes and measured setup
requirements. The bootstrapping and live phases can use the same persistent
layout; live experience can continue to repair its relations through `observe`.

## Combine routine layers and recursive observation

An action need not come from the highest observer. Here ordinary layers produce
the answer while an observer reads their states and errors. Its relations can
return influence to those same states through joint settlement. No second body
interface, separate evaluator or call to the observer is needed.

```python
mixed = Cortex(seed=7)
signal = mixed.input("signal", shape=1)
representation = mixed.column("representation", patches=4, inputs=signal)
integration = mixed.column("integration", patches=2, inputs=representation)
response = mixed.column("response", patches=1, inputs=integration)
reflection = mixed.observer(
    "reflection", patches=2, observes=(integration, response),
)
mixed.output("answer", shape=1, reads=response)
mixed_brain = mixed.build()

# Reuse the same examples and checks from the shared teaching loop above.
report = bootstrap(
    mixed_brain, examples, checks=checks, epochs=20,
    max_error=0.1, batch_size=4, seed=2,
)
assert report["passed"], report
result = mixed_brain.settle({"signal": [0.5]})
assert result["qualified"]
assert abs(result["outputs"]["answer"][0] - 0.3) < 0.1
```

This checks that a mixed graph learns the small relation. It does not show that
the observer helps, sleeps during routine work or runs at its own speed. To test
whether it helps, train a capable ordinary control with the same information
and account for parameters, acquisition work and complete query cost. See
[System 1 and System 2](BRAIN_DESIGN.md) for the behavioral goal and current
runtime boundary.

## Combine sensory branches

```python
from cadence import Cortex

cortex = Cortex(seed=7)
eyes = cortex.input("eyes", shape=(4, 4))
ears = cortex.input("ears", shape=(4,))
vision = cortex.column("vision", patches=8, inputs=eyes)
hearing = cortex.column("hearing", patches=8, inputs=ears)
fusion = cortex.column("fusion", patches=8, inputs=(vision, hearing))
reflection = cortex.observer(
    "reflection", patches=8, observes=(vision, hearing, fusion),
)
meta = cortex.observer("meta", patches=4, observes=(fusion, reflection))
cortex.output("move", shape=2, reads=meta)
brain = cortex.build()
assert brain.inspect()["patches"] == 36
assert brain.inspect()["sensor_coverage"] == 20
```

## Connectivity and cost

The default `fan_in=None` connects each patch to every coordinate of its declared
sources. Set a positive integer, such as `Cortex(fan_in=8)`, to request sparse
wiring. That count is capped at source width and raised when needed for aggregate
coverage across the destination population. Aggregate coverage does not guarantee
that each selected output can use every sample: unconnected patches cannot relay
what they read through shared, fixed sensors.

Observation has both state and error ports. Sharing a state port through `inputs`
and `observes` does not duplicate that connection. Dense wiring increases cost;
256 patches reading a 240×320 RGB image require 58,982,400 input connections,
exceeding the default one-million limit. Use a smaller declared representation,
explicit sparsity or a larger budget; the builder never silently drops inputs to
fit. Neither wiring choice supplies learned visual features by itself.

`max_inputs`, `max_patches` and `max_connections` bound compilation. The solver's
budget counts accepted repair sweeps; work also includes initial evaluations,
rejected line-search proposals and final qualification. Defaults are starting
points for small models, not a guarantee of interactive speed at arbitrary
width or sensor resolution. Optional tensor execution can parallelize patch
arithmetic and private experience rows in `observe_batch`; it does not change
population width or observation depth. Batch size is a separate learning and
memory-cost choice, not additional brain capacity. See
[batch execution](ACCELERATION.md#batch-experience-on-one-device).

To test the value of recursion, compare a nested layout with shallow and
ordinary-composition controls, match information and capacity, and account for
all bootstrapping, live learning and settlement work. Measure target-free task
performance, numerical refusal rates and latency. A wider brain or a larger
budget alone is not proof that observation adds value.

Test direct sensory and action-history paths separately from latent-state and
prediction-error readback. A policy can use a bypass while its observers remain
numerically active. Conversely, removing bypasses can force a latent path
without teaching it useful perception. Perturb sensory inputs with history
held fixed, and compare ordinary state connections with state-and-error
observation. Detached lesions measure sensitivity; an action changing under a
lesion does not show that the intact path helps the task. Check qualified
environment behavior as well as scores, and distinguish lesions from trained
controls. Equal patch counts need not imply equal parameters, connections or
work.
