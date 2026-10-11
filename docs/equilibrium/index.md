# Cadence documentation

This advanced guide covers `cadence.experimental.equilibrium`: one connected
graph of patches that repair disagreement together. Choose population sizes and
the ports they read, teach through the same joint repair, then measure free
behavior. Learned relations persist and remain plastic, and outputs are selected
patch states from a qualified solve, not answers from a separate readout network.

These experiments keep their own energy and stationarity equations and their own
explicit history; they do not stand in for the default `Brain.compose`
implementation, which [one continuing equilibrium brain](../world-model.md)
describes. Small independent calibrations are mechanism controls.

## A short learning path

1. [Quickstart](QUICKSTART.md): construct a brain, teach a relation, test fresh
   predictions without targets, then save and resume.
2. [Wiring examples](VARIANTS.md): add states, intermediate populations, branches
   or optional error readback within the same graph.
3. [Brain design](BRAIN_DESIGN.md): choose observations, temporal context,
   connected capacity and a cost budget.
4. [Bootstrapping](BOOTSTRAP.md), then [live operation](LIVE.md): acquire a
   behavior, execute it, learn from actual outcomes and check retained skills.

Bootstrapping and live operation use the same brain and learning rule. Numerical
qualification and useful behavior need separate checks. The
[experimental boundary](EXPERIMENTAL.md) explains recursive observation, intended
routine/correction roles and their current limits.

## Reference and examples

| Read | Purpose |
| --- | --- |
| [Runnable examples](../../examples/equilibrium/README.md) | Learning, body control, explicit history, reward, batching and cost measurement |
| [Agent recipe](AGENTS.md) | Application workflow and documentation rules |
| [GPU execution and parallel experience](ACCELERATION.md) | Devices, precision, batch repair and independent lives |
| [Architecture guide](DRSN.md) | Patch equations, connected populations and recursive readback |
| [Capability and cost](PERFORMANCE.md) | Query and learning work and comparison controls |
| [Processing patch](ELEMENT.md) | Local state, prediction errors, repair and retained relations |
| [API reference](REFERENCE.md) | Classes, parameters, methods and diagnostics |
| [Specification](SPECIFICATION.md) | Qualification, refusal, witness admission and saved continuation |

[Source](https://github.com/muellerberndt/cadence) ·
[Public demos](https://github.com/muellerberndt/cadence-demos) ·
[Release notes](../../CHANGELOG.md)
