# Cadence documentation

Welcome. This page is the way in: what Cadence is, how to install it, a first brain in ten lines,
the handful of ideas everything else builds on, and a map of the guides.

## Cadence in one minute

Cadence is a Python library for building brains that learn while they live. A brain senses an
observation, **settles** (its neurons drive each other until their activity comes to rest), reads
an action off the settled state, and learns from what that action brings.

Animal and human brains learn from experience and not by backpropagation with gradient descent.
Cadence is designed the same way: its connections change locally while the brain keeps running.

There is no separate training phase and no inference mode: the brain you train is the brain that
acts, and it keeps its memories from one moment to the next. The default brain, **System 1**,
has a processing region, a working trace, a recall memory of recent moments and an associative
memory of what paid. Optional **System 2** observer regions settle in the same brain and add
recursive feedback.

Cadence is research software. It is a good fit if you want to study or build agents that learn
continuously from experience; [Cadence for machine-learning readers](../README.md#for-machine-learning-readers)
compares it with networks trained by backpropagation.

## Install

```sh
python -m pip install cadence-net==0.98.0
```

Cadence needs Python 3.11 or later and NumPy. Optional extras add speed and hardware:

| Extra | Adds | Use it for |
| --- | --- | --- |
| `[fast]` | Numba and SciPy | faster CPU work; recommended |
| `[accel]` | PyTorch | NVIDIA GPUs (CUDA) |
| `[apple]` | MLX | Apple silicon GPUs |

For example `python -m pip install "cadence-net[fast]==0.98.0"`. From a checkout of the
repository, run `python -m pip install -e .` at its root. A new brain picks the fastest backend
the machine has for its size by itself ([backends](backends.md)). The
[v0.98.0 documentation](https://github.com/muellerberndt/cadence/blob/v0.98.0/docs/README.md)
is pinned to the release tag.

## Your first brain

<!-- not-run: this block runs as the first example of start.md -->
```python
import numpy as np
from cadence import Brain

brain = Brain.compose(inputs=4, actions=2, arousal=True)  # 4 senses, 2 possible actions
observation = np.array([[1.0, 0.0, 0.0, 0.0]])            # what it senses at this moment
action = brain.live(observation)                          # it settles and chooses an action
reward = (action == 0).astype(float)                      # your world scores that action
next_observation = np.array([[0.0, 1.0, 0.0, 0.0]])
action = brain.live(next_observation, reward=reward)      # it learns from the score, then acts again
assert brain.learner.updates > 0
```

[Start here](start.md) takes this example apart, line by line.

## The ideas, in the order you need them

Each idea builds on the one before. [Start here](start.md) shows each with a runnable example.

1. **Observation and action.** An observation is a row of numbers, what the brain senses at one
   moment; an action is one of a fixed set of choices, an integer
   ([observations, actions and streams](start.md#2-observations-actions-and-streams)).
2. **Settle.** The brain answers by letting its neurons drive each other until their activity
   comes to rest. The settled state is the brain's interpretation, and the action is read from it
   ([settling](start.md#3-settling-how-a-brain-answers)).
3. **Act.** `act`, `step` and `live` return actions. A sampled action waits for its outcome; a
   greedy one does not ([which call](start.md#5-which-call-should-i-use)).
4. **Lesson or reward.** A lesson shows the right answer; a reward scores the brain's own action.
   Both change the connections locally
   ([two ways to learn](start.md#4-two-ways-to-learn-lessons-and-outcomes)).
5. **The stream state.** Between moments a brain carries its activity, its working trace, its
   recall slots and any action awaiting an outcome. That is how what came before shapes the
   present ([the stream state](start.md#6-the-stream-state-what-carries-from-moment-to-moment)).
6. **Reset and save.** `reset()` starts new streams and keeps what was learned; `save` and
   `Brain.load` continue the same life later
   ([reset](start.md#7-reset-start-a-new-stream), [save](start.md#8-save-and-load)).

Then, in this order:

1. [Parts of a brain](modules.md): the catalogue of every part a brain can be built from, five
   whole brains composed for a purpose, and the modules that arrive as files.
2. [Train a brain](teach.md): lessons, rewards, distillation, rare classes, curricula, rates,
   evaluation and what to do when training goes wrong.
3. [Memory](memory.md): which memory serves which task, with a minimal example of each.
4. [Learning from reward](reward.md): eligibility traces, the actor's rates, delayed credit,
   maturity.
5. [Tune and read a living brain](tune.md): every setting, `describe`, `retune`, larger brains.
6. [API reference](api.md): every public name with its signature and refusals.

## Words used in these docs

| Word | Meaning |
| --- | --- |
| moment | one call that senses one observation per stream and usually answers with an action |
| stream | one continuing life, one row of every batch; it lasts until `reset()` |
| settle, sweep | relaxing the neurons to a resting state under the observation; one round of updates |
| residual, qualified | how far a state is from satisfying the equations; a state close enough to act from |
| lesson | teaching the right answer for an observation (`lesson`, `step(teacher=...)`) |
| outcome, reward | the world's score of an action, delivered on the next call |
| greedy, sampled | the most active action; an action drawn at random by preference, which can be credited |
| setting, gene | a named value of `compose` or `retune`, such as `actor_eta` |
| founder | a setting's original value; a new brain can hold another, as the guides say |
| System 1, System 2 | the default continuing brain; optional observer regions in the same settle |

<a id="by-situation"></a>

## Find the guide for your task

| You want | Use | Page |
| --- | --- | --- |
| To choose the parts of a brain for a purpose: a transcriber, an agent that remembers facts, a body that reaches | the catalogue of parts, each with how it joins a brain | [Parts of a brain](modules.md) |
| To train on a dataset, a teacher, rewards or another brain | `lesson`, `step`, `targets=`, `weights=` | [Train a brain](teach.md) |
| The world to pay for good actions | `step(reward=...)`, `live`, `actor_eta` | [Learning from reward](reward.md) |
| A memory for a task | the comparison table | [Memory](memory.md) |
| Wide inputs: images, codes | image `inputs`, `code=units`, `learned_code=units` | [Wide inputs](senses.md) |
| A speech recogniser from recordings and transcripts, adapted to a speaker, on live audio | `cadence.speech` (opt-in) | [Speech](speech.md) |
| A map of where things are while they are out of view, by kind and by place | `cadence.space` (opt-in) | [Space](space.md) |
| To read or change a living brain | `describe`, `retune`, `last_settlement` | [Tune and read a living brain](tune.md) |
| A deep or wide brain | `modules=(64, 32, 32)`, `apical=0.25`, `teaching="deep"` | [Larger brains](tune.md#larger-brains) |
| A GPU, another device or precision | `backend`, `device`, `precision` | [Backends](backends.md) |
| A brain that keeps learning after it has acquired a skill | `maturity=True` | [Maturity](reward.md#maturity-opt-in) |
| To give a trained brain a memory, more outputs or another brain's regions | `grow`, `graft` | [Grow and combine brains](grow.md) |
| A module that arrives as a file: the built-in map of things, or a trained brain's regions of your own | `cd.modules.load("place")`, `cd.load_module(path)`, `brain.space`, `save_module`, `trained=[module]`, `graft(module)` | [Modules](modules.md#modules-parts-that-arrive-as-files) |
| To show a result | baselines, controls, receipts | [Evaluate a brain](evaluate.md) |
| To design a continuing application or a behavioural claim | observation and action boundaries, acquisition and recovery | [Task design](task-design.md), [the design direction](world-model.md) |
| Your own regions, a connectome, evolution | `Genome`, `develop`, `compose(connectome=)`, `evolve` | [Your own anatomy](anatomy.md) |
| How the graph settles and learns, in full | the neuron, the free/nudged rule, the certificate | [How the brain settles](settling.md) |
| A symptom remedied | symptom to remedy | [Troubleshooting](troubleshooting.md) |
| The guarantee of each model family | the numerical and learning contracts | [Contracts](contracts.md) |

## Evaluate and contribute

Design a claim with [evaluation](evaluate.md), test it with
[the protocol instrument](../benchmarks/instruments/README.md) and report it with
[receipts](evaluate.md#receipts). [CONTRIBUTING.md](../CONTRIBUTING.md) lists the checks,
[CHANGELOG.md](../CHANGELOG.md) the releases, the [manifesto](../MANIFESTO.md) the aims,
[AGENTS.md](../AGENTS.md) the construction rules, the [paper](https://philpapers.org/rec/MUECAP-2)
the hypothesis and [cadence-demos](https://github.com/muellerberndt/cadence-demos) the
application demos.
