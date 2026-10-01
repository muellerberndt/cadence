# Cadence 0.60 documentation

This guide covers **`0.60.0.dev0` on `main`**. Build a learned routine with a
flat population, add ordinary deep layers for intermediate representations,
and test recursive observers when state-and-error feedback could help. Every
layout uses the same bounded patch rule, settlement and learning API.

| You need | Start with |
| --- | --- |
| A small direct response | A flat population reading the sensors |
| A more expressive routine | Ordinary deep layers whose live states settle together |
| Internal error readback | Recursive observers in the same jointly settling brain |
| Both routine layers and observers | A mixed layout with one external observation/action/outcome interface |

These are layout choices, not latency guarantees. **System 1** means acquired
routine competence, which can be deep. **System 2** means additional recursive
correction when routine behavior fails. The intended cycle is routine →
disturbance → useful correction → cheaper learned routine. Automatic internal
attention and independently progressing fast/slow populations remain development
requirements; adding an observer does not enable them. See the
[0.60 migration and capability boundary](MIGRATION_060.md).

## A short learning path

1. [Quickstart](QUICKSTART.md): construct a brain, bootstrap a relation, test
   fresh predictions without targets, then save and resume.
2. [Layout quickstarts](VARIANTS.md): build flat, deep, recursive and mixed
   layouts without changing the external interface.
3. [Brain design](BRAIN_DESIGN.md): choose observations, temporal context,
   connected capacity and a cost budget for the task you actually need.
4. [Bootstrapping](BOOTSTRAP.md), then [live operation](LIVE.md): acquire a
   behavior, execute it, learn from actual outcomes and check old skills.

Bootstrapping and live operation are lifecycle phases, separate from System 1
and System 2. The same brain can continue learning in both phases. Numerical
settlement and useful behavior need separate checks.

## Guides and runnable examples

| Read | Purpose |
| --- | --- |
| [Runnable examples](../examples/README.md) | Layout learning, a learned body controller, explicit history, reward, batching and cost measurement |
| [Agent recipe](AGENTS.md) | A compact integration workflow and rules for accurate architecture claims |
| [GPU execution and parallel experience](ACCELERATION.md) | Devices, precision, batch repair and independent lives |
| [Architecture guide](DRSN.md) | Patch equations, multimodal branches, ordinary coupling and recursive readback |
| [Depth, latency and useful work](PERFORMANCE.md) | Query and learning costs, comparison controls and versioned demo evidence |
| [Processing patch](ELEMENT.md) | Local state, prediction errors, repair and retained relations |
| [API reference](REFERENCE.md) | Public classes, parameters, methods and diagnostics |
| [Specification](SPECIFICATION.md) | Qualification, refusal, witness admission and saved continuation |
| [Migrating to 0.60](MIGRATION_060.md) | Installation, changed contracts, source-bound checkpoints and demo reproduction |

[Source](https://github.com/muellerberndt/cadence) ·
[Public demos](https://github.com/muellerberndt/cadence-demos) ·
[Release notes](../CHANGELOG.md)
