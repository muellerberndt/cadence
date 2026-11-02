# Cadence documentation

Cadence builds an animal-like brain: one continuing equilibrium that settles each observation
into an interpretation, acts on it and repairs its local relationships from the outcome.
System 1 is the default brain, System 2 observers are optional regions in the same settlement,
and an MLP or classifier setup is a control to compare against. The
[manifesto](../MANIFESTO.md) states the aims and [AGENTS.md](../AGENTS.md) the construction rules.

## Install

```sh
python -m pip install cadence-net==0.92.1
```

Cadence needs Python 3.11 or later and NumPy. From a checkout, run `python -m pip install -e .`
at the repository root. The extras `[fast]` (Numba and SciPy), `[accel]` (PyTorch) and
`[apple]` (MLX) add backends; [backends](backends.md) says which to choose. The
[v0.92.1 documentation](https://github.com/muellerberndt/cadence/blob/v0.92.1/docs/README.md)
is pinned to the release tag.

## By situation

| You want | Use | Page |
| --- | --- | --- |
| A creature in a body that lives, acts and learns | `Brain.compose(inputs, actions, arousal=True)`, `live`, `wait`, `reset`, `save`, `Brain.load` | [Start a continuing brain](start.md) |
| To teach by demonstration | `lesson`, `step(teacher=)`, `teaching=` | [Teach by demonstration](teach.md) |
| The world to pay for good acts | `step`, `act` and `learn`, delayed pay, chains | [Learning from reward](reward.md) |
| The brain to remember | Start with the default trace/store; add `memory_choice=True` for episodic voting or `learned_code=units, attention=True` for contextual retrieval | [Choose a memory](memory.md#start-with-the-need) |
| Wide inputs: images, codes | image `inputs`, `code=units`, `learned_code=units` | [Wide inputs](senses.md) |
| A deep brain, with every memory and attention feature | `modules=(64, 32, 32)`, `apical=0.25`, the depth teaching settings, `attention=True`, `recall_heads=2` | [Deep brains](tune.md#deep-brains) |
| To read or change a living brain | `describe`, `retune`, `last_settlement`, the founders | [Tune and read](tune.md) |
| To give a trained brain a memory, more heads or more outputs, or to put trained regions into a larger brain | `grow(recall=True)`, `grow(slots=(4, 2))`, `graft(donor, regions)`, `graft(donor, into=...)` | [Grow and combine brains](grow.md) |
| To show a result | baselines, controls, `fit` as a control, receipts | [Evaluate a brain](evaluate.md) |
| To design a behavioral claim or a continuing application | observation/action boundaries, context interventions, acquisition and recovery | [Task design](task-design.md), [world-model scope](world-model.md) |
| Your own regions, a connectome, evolution | `Genome`, `develop`, `compose(connectome=)`, `evolve` | [Your own anatomy](anatomy.md) |
| To know how the graph settles and learns | the neuron, the free/nudged rule, the certificate | [How the brain settles](settling.md) |
| Another device or precision | backends, devices, extras | [Backends](backends.md) |
| A symptom remedied | symptom to remedy | [Troubleshooting](troubleshooting.md) |
| The guarantee of each model family | the numerical and learning contracts | [Contracts](contracts.md) |
| Every public name | signatures and refusals | [API reference](api.md) |

Each model family carries its own guarantee, and one family's guarantee does not certify a
composition of families ([contracts](contracts.md)). Readers who know machine learning find
the comparison in the [README](../README.md#for-machine-learning-readers).

## Evaluate and contribute

Design a claim with [evaluation](evaluate.md), test it with
[the protocol instrument](../benchmarks/instruments/README.md) and report it with
[receipts](evaluate.md#receipts). [CONTRIBUTING.md](../CONTRIBUTING.md) lists the checks,
[CHANGELOG.md](../CHANGELOG.md) the releases, the [paper](https://philpapers.org/rec/MUECAP-2)
the hypothesis and [cadence-demos](https://github.com/muellerberndt/cadence-demos) the
application demos.
