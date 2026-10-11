# Advanced population experiments

This subtree documents `cadence.experimental.equilibrium`, a separate
experimental population model for studying exact prediction-error feedback.
Start with [the catalogue and reading path](index.md).

For the default brain, read [one continuing equilibrium brain](../world-model.md)
and [the main brain guide](../brain.md) instead. The population model's joint
stationarity law stays distinct from the default neural graph, it uses explicit
`History` rather than Brain's trace and associative memory, and it has no
private planner.

```python
from cadence.experimental.equilibrium import Cortex
```

Each bounded patch has state, ports, retained relations and a local prediction.
Connected patches repair their shared state: a population reads live states with
`inputs=`, or also reads exact current errors with `observes=`, and returning
influence participates in the same energy and qualification check.

The package is experimental. APIs and saved formats may change, and there is no
cross-version compatibility commitment, so save the source and the complete
application state for experiments you need to reproduce.
