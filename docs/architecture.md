# Architecture and integration

Cadence models a continuing brain through bounded local state, ports, readback,
records and plastic connections. **System 1** is the default observation/action
loop with memory and learning. **System 2** optionally adds recursive observer
regions within that same neural graph. Local disagreement repair seeks a coherent
state; actual consequences determine whether its behavior was useful.

The [world-model guide](world-model.md) is the canonical application frame:
bootstrap reusable relations and memory, operate with them, repair witnessed
failures and resume the same brain. Learned relationships support a family of
equilibria under changing evidence. The separate implementations below retain
their own contracts; listing their mechanisms does not establish an integrated
world model or automatic failure-gated cheap operation.

## One continuing brain

`Brain.compose` connects sensory, processing, motor and memory regions.
Base modules form reciprocal paths. Optional observers read and return to base,
motor and earlier observer regions; they do not run as a separate answer
controller. A graph can be deep and modular without observers.

```python
import numpy as np
from cadence import Brain

brain = Brain.compose(inputs=4, actions=2, arousal=True)
reading = np.array([[1.0, 0.0, 0.0, 0.0]])
action = brain.live(reading)
assert action.shape == (1,)
```

All live neural coordinates must meet the declared full-equation tolerance
before an action is issued. Stored traces and associative records supply
explicit boundary information; they are not silently frozen unresolved live
coordinates. Finite free/nudged learning phases retain their own contracts.
Qualified free solves may use numerical damping inside one budget and then
check the original model. See [contracts](contracts.md).

`live` receives the preceding action's actual outcome and acts on the current
observation, with arousal deciding when to explore and learn. Use `step` for
explicit teaching, batches or learning from every outcome.
`imagine` privately settles supplied observations with a copied
trace. It does not learn from predictions or model environmental consequences
by itself. Saved continuation includes pending feedback and memories; the body
is saved separately.

## Choose the operation the task needs

| Operation | Interface | Meaning |
| --- | --- | --- |
| Continuing perception, action and reward learning | `Brain.live` | One stream with arousal; `step` supports explicit teaching, batches and learning from every outcome |
| Earlier activity affecting later answers | `Trace`, `Afterglow` | Retained temporal input with declared decay and optional movement weighting |
| Fast and persistent associations | `SynapticMemory` | Observed key/value updates with finite capacity and possible interference |
| Individual event records and consolidation | `RecordPatchNet` | Context, learned parameters, writable records and `sleep` |
| Learned environmental dynamics | `TemporalPatchNet` | Relations over finite observed paths |
| Private action planning | `TemporalPatchNet.plan` | Continuous input proposals under a learned model and supplied goals |
| Protection of selected responses | `TemporalMemory` | Finite selected response subspaces constrain later learning |

These operations share the aim of local readback and repair, but have distinct
mathematical contracts. Graph and temporal learners use equilibrium contrasts;
record and belief models also use explicit adjoints and record writes.
Sequential record lookup or steering is not a joint neural-equilibrium
certificate. The advanced [population solver](equilibrium/index.md) has exact
state-and-error readback under a separate model.

## A temporal learning and planning model

Use the temporal API when the task requires learned consequences of actions.
Its private planner holds learned parameters and actual observations fixed,
repairs proposed continuous inputs and checks target-free predictions before
acceptance. Executing the proposed action and measuring its outcome are separate
body operations.

```python
from cadence import TemporalPatchNet, TemporalMemory

net = TemporalPatchNet(2, 8, 1, seed=151)
memory = TemporalMemory()
heard = np.array([[[1.0, 0.0]]])
net.reset()
assert net.observe(heard, np.array([[[0.2]]])).updated

# Explicitly protect this response from cold context.
memory.protect(net, heard, state=np.zeros((1, 8)))
net.reset()
assert memory.observe(net, np.array([[[0.0, 1.0]]]), np.array([[[0.3]]])).updated
net.reset()
assert net.advance(heard).converged
assert net.imagine(np.zeros((1, 4, 2))).converged
```

Only free activity becomes live after learning; target-detuned states remain
private. Protection preserves selected current responses, not an unobserved
truth. Supplied goals and importance, finite capacity, changing cues and
interference remain part of the task contract. The
[memory/planning example](../examples/memory_imagination.py) checks a bounded
case against actual toy-body outcomes.

## Evaluate the behavior

Adapters define sensory meanings, executable actions and teaching access.
A lossy sensory summary may omit distinctions needed to act. Test acquisition,
free recall, changes in goals, competing learning, private-state isolation and
saved continuation. A settled state can still describe a poor predictor; useful
recursive correction and human-like flexibility require behavioral evidence.

[Build a brain](brain.md) · [Continuous interaction](continuous.md) ·
[Temporal learning](temporal.md) · [Planning](planning.md) ·
[Response protection](temporal-memory.md) ·
[Paper](https://philpapers.org/rec/MUECAP-2)
