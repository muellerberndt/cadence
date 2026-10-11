# Start a continuing brain

Build one **System 1** brain with `Brain.compose`, then run it with `brain.live`.

```python
from cadence import Brain

brain = Brain.compose(inputs=4, actions=2, arousal=True)
action = brain.live([[1.0, 0.0, 0.0, 0.0]])
```

It includes connected processing regions, motor choices, a working trace and
fast/persistent associative memory. Regions
carry local state, exchange signals and repair disagreement in one neural
settlement. Optional **System 2** adds observing regions with returning feedback
in that same graph.

The [world-model guide](world-model.md) explains the intended lifecycle:
bootstrap a useful interpretation, use it, repair witnessed failures and continue
the same brain. This quickstart exercises equilibrium action and memory;
it does not yet integrate learned environmental transitions.

Python 3.11+ and NumPy are required. Install Cadence 0.81.0:

```bash
python -m pip install cadence-net==0.81.0
```

## Observe, act and learn

```python
import numpy as np

observation = np.array([[1.0, 0.0, 0.0, 0.0]])

# Execute the choice in a tiny environment: action 0 earns one unit.
reward = (action == 0).astype(float)
next_observation = np.array([[0.0, 1.0, 0.0, 0.0]])
action = brain.live(next_observation, reward=reward)
assert action.shape == (1,)
assert brain.learner.updates > 0
```

`live` follows one continuing stream: pass one observation row and execute its
returned action index. The constructor uses the default System 1 layout and
enables arousal. A young or aroused brain explores and learns; a calm brain
answers greedily without learning. Report the **previous action's actual outcome**
before requesting the next action. Use `reset()` to start a different stream.

For batches, supplied teaching labels or learning from every outcome, use
[`step`](continuous.md). It is the explicit learning loop for those tasks.
Use named overrides such as `actor_eta=0.1, actor_eta_bias=0.01` only after
checking the task; `brain.retune(...)` changes selected settings without losing
acquired state. [`brain.describe()` and the defaults table](brain.md#defaults-and-expert-overrides)
show what is running and which operation uses each rate.

This short example exercises a feedback update, not a learned policy benchmark.
Memory and learned associations can affect later choices; capacity is finite
and memories can interfere. [Continuous interaction](continuous.md) covers
teaching, episodes and memory timing.

## Inspect the work of answering

```python
report = brain.last_settlement
assert report is not None and report["qualified"]
assert report["max_residual"] <= report["tolerance"]
print("free-answer sweeps:", report["steps"], "residual:", report["max_residual"])
```

The read-only report records the latest free-answer solve, including a refused
attempt. It also counts residual checks and numerical damping. It excludes
reward eligibility, teaching, feedback and memory work. Compare these counters
with actual task outcomes; small residual means internal consistency, not a
correct action, predictable environment or low physical energy.

The [continuing example](../examples/continuing_brain.py) measures these signals
through bootstrap, unchanged conditions, disruption and correction. It supplies
a corrective teacher only after an executed mistake, repeating the failed cue
so that the teacher labels the current observation. It still consumes every
real reward exactly once. This is an explicit teaching policy in the example;
`step` does not automatically suppress successful-outcome learning.

For narrower existing mechanisms, see [centered dopamine and selective activity](reward.md#centered-dopamine-and-selective-activity).
Reducing an update does not necessarily reduce the work spent computing it.

## Imagine privately and resume

```python
phases = brain.imagine([observation, next_observation])
assert len(phases) == 2 and all(np.all(phase.qualified) for phase in phases)

brain.save("brain.npz")
resumed = Brain.load("brain.npz")

# Resume the same pending action with the same measured outcome.
reward = (action == 0).astype(float)
continued = brain.live(observation, reward=reward)
replayed = resumed.live(observation, reward=reward)
assert np.array_equal(continued, replayed)
assert brain.arousal.to_dict() == resumed.arousal.to_dict()
```

`imagine` carries a private trace through supplied observations. It leaves live
memory, parameters, random state and pending feedback unchanged. Inspect every
phase's `qualified` flags: the result stops at the first refused phase. This
evaluates the brain's responses to supplied observations; a learned model of
environmental consequences belongs to [temporal planning](planning.md).

Save/load preserves the current brain's full continuation, including pending
feedback. Save the environment separately and resume its stream identities too.

## Add optional observers

Optional System 2 extends a two-module System 1 with observers:

```python
recursive = Brain.compose(
    inputs=4, actions=2, modules=(16, 8), observers=(8,), seed=7,
    arousal=True,
)
assert recursive.live(observation).shape == (1,)
```

Observers read and return influence to processing regions, motor regions and
earlier observers. They participate in the same settlement and interaction API.
This provides recursive feedback; a useful task advantage must be learned and
measured.

Actions and independent predictions require the full neural equation residual
to meet the configured tolerance. If `live` refuses its forecast before accepting
feedback, the same call can be retried. If it accepts the outcome and then refuses
the next action, retry `live(observation)` without submitting the outcome again.
See [routine and repair](continuous.md#routine-and-repair-live).

## Specialist guides

- [Compose a brain](brain.md): custom regions, wiring and checkpoint semantics.
- [Memory](memory.md): working traces, fast associations and consolidation.
- [Temporal models](temporal.md) and [response protection](temporal-memory.md):
  learned consequences, private planning and selected durable responses.
- [Record patches](record-patch.md): one-write event records, categorical ports
  and learning from stored completions.
- [Reciprocal patches](patchnet.md): explicit patch ports and local contrasts.
- [The population solver](equilibrium/index.md): advanced exact state-and-error
  readback under its own numerical contract.
