# Experimental fixed connectivity

Pin a release or a commit when reproducing an experiment with
`cadence.experimental.partitioned`.

`PartitionedTemporalPatchNet` applies fixed boolean connection masks to one
`TemporalPatchNet`. Coordinate groups remain parts of a single jointly repaired
temporal path, with local state, ports, readback and detuning. This is a supplied
architecture for controlled experiments. It does not discover modules, assign
their roles, learn specialization or change the default temporal model.

## What the masks mean

The existing causal recurrence is unchanged:

```text
h[t] = A @ tanh(h[t-1]) + B @ input[t]
y[t] = C @ tanh(h[t])
```

The masks have the same shapes as `A`, `B` and `C`: `(hidden, hidden)`,
`(hidden, inputs)` and `(outputs, hidden)`. An `A[row, column]` entry permits a
connection from the preceding state's column to the current state's row.
Recurrent communication therefore spans one represented path step. Its physical
duration, if any, belongs to the application; it is not one solver iteration.
There is no added instantaneous coupling or independent component clock.

Initialization first draws the ordinary dense parameters, then zeros forbidden
entries. It does **not** renormalize the masked recurrence. `initial_radius`
describes the dense initialization before masking, not the resulting radius.
All-true masks reproduce the dense model's numerical phases and updates.

The local parameter gradients are masked before any candidate update or
parameter backtracking. `set_parameters` rejects nonzero forbidden entries.
`masks` returns detached copies; changing a copy cannot change the architecture.
The masks are immutable for the lifetime of the model.

## A small executable example

The optional `two_group_masks` helper declares two coordinate groups. Its names
`context_hidden` and `motor_hidden` describe supplied roles: both groups use the
same dynamics. Within-group recurrence is dense; `cross_coupling=True` permits
both delayed directions. Only the second group has direct output connections.
Input sets may overlap or be empty. For another topology, supply your own three
boolean arrays instead.

```python
import numpy as np

from cadence.experimental import PartitionedTemporalPatchNet, two_group_masks

masks = two_group_masks(
    context_hidden=3, motor_hidden=3, inputs=2, outputs=1,
    context_inputs=[0], motor_inputs=[1], cross_coupling=True,
)
net = PartitionedTemporalPatchNet(2, 6, 1, masks=masks, seed=7)
inputs = np.zeros((1, 4, 2))
inputs[0, 0, 0] = 1.0  # one observed context pulse
inputs[0, :, 1] = [0.0, 0.1, -0.1, 0.2]
target = np.array([[[0.0], [0.1], [0.2], [0.1]]])

result = net.observe(inputs, target, beta=0.01, rate=0.1, backtrack=True)
assert result.free.converged
for key, values in net.parameters().items():
    assert np.all(values[~net.masks[key]] == 0)

before = net.snapshot()
prediction = net.imagine(inputs)
restored = PartitionedTemporalPatchNet.restore(before)
np.testing.assert_array_equal(restored.imagine(inputs).output, prediction.output)
assert all(np.array_equal(before[key], net.snapshot()[key]) for key in before)
```

`observe`, `advance`, `imagine`, `plan`, `reset` and `readback` retain the
[temporal model's contracts](temporal.md). In particular, private planning
adjusts selected input ports under fixed parameters; it does not execute an
action or alter the masks. Inspect returned phase and learning outcomes rather
than assuming that a permitted connection guarantees acquisition.

## Checkpoints and compatibility

Snapshots store the masks alongside the complete temporal continuation.
Use `PartitionedTemporalPatchNet.restore(snapshot)` or
`PartitionedTemporalPatchNet.load(path)` after `net.save(path)`. Restoring through
the base `TemporalPatchNet` class discards future mask enforcement and is not a
substitute. The subclass restore validates both the masks and the zero values
of forbidden parameters before returning a model.

**`TemporalMemory.observe` rejects this subclass with `TypeError` before
changing the network or memory.** `TemporalMemory.protect` can collect response
constraints, but this does not enable a combined protected-learning transaction.
Neither ordinary projection nor its optional readout metric supports this
combination. Do not bypass the rejection by restoring a base-class model.

## What has been checked

Tests cover dense parity, restricted finite-difference gradients, legal
backtracked updates, failed-step behavior, checkpoint continuation, private
planning and atomic rejection by `TemporalMemory`. A small hash-bound fixture
in `tests/fixtures/partitioned_phrase/` checks a real development update against
the original helper, including detuned endpoints and committed parameters.

These implementation checks do not establish a behavioral advantage. In the
bounded four-bar recall comparison that supplied the fixture, the manually
partitioned model recalled fewer event identities than the same-width dense
control; both runs stopped at the shared computation cap. No learned
specialization or original composition was demonstrated.

Masks reduce the number of trainable entries, not the allocated arrays or the
dense hidden-width message solver. Count boolean masks, parameters, phase work
and initialization changes when comparing architectures. See
[scaling](scaling.md) and [common missteps](missteps.md).
