<p align="center">
  <img src="https://raw.githubusercontent.com/muellerberndt/cadence/main/docs/assets/patchnet.svg" alt="A patch net: owners hold state, seams carry it both ways, every owner repairs its own patch until the net is at rest" width="100%">
</p>

<h1 align="center">Cadence</h1>

<p align="center"><strong>Machine learning by patch-net settlement.</strong><br>
Owner-local repair, no backward pass, held-out tests, receipts.</p>

<p align="center">
  <a href="https://pypi.org/project/cadence-net/"><img alt="PyPI" src="https://img.shields.io/pypi/v/cadence-net?color=1f8a70&label=cadence-net"></a>
  <a href="https://github.com/muellerberndt/cadence/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/muellerberndt/cadence/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-3d5a80">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-9fb3c8">
</p>

<p align="center">
  <a href="docs/index.md">Docs</a> ·
  <a href="docs/quickstart.md">Quickstart</a> ·
  <a href="docs/learning.md">How it learns</a> ·
  <a href="https://github.com/muellerberndt/cadence-examples">Examples</a> ·
  <a href="https://claude.ai/code/artifact/ee7a8b53-be8c-4c34-9f91-43d6eaf77be8">Play the demos</a>
</p>

---

A **patch net** is a set of *owners*, each holding one patch of state, joined by declared
*seams*. Nothing is computed globally. Every owner repairs its own patch from what arrives
over its seams, and the state the net comes to rest in is the answer. Learning is the same
settlement run again with the outputs nudged: every seam moves on what its own two ends
did. Cadence is the library for building, settling, training, testing, and certifying
such nets, from a six-owner toy to a 161,827-owner nervous system read from a connectome.

```python
import cadence as cd

wiring = cd.layered(64, 32, 10, density=1.0, seed=0)        # input, hidden, output owners
learner = cd.Learner(cd.Settlement(wiring, cd.learning_rule()), wiring.sets["output"],
                     cd.LearnerConfig(eta=3.0, beta=0.1, temperature=0.1))
for idx in batches:
    learner.step(drive[idx], labels[idx])                    # settle free, settle nudged, update locally
learner.accuracy(drive_test, labels_test)                    # 0.96 on the 8x8 digits
```

## Why Cadence

- **One rule for answering and learning.** A settlement makes the prediction; a nudged
  settlement teaches. There is no forward pass, no backward pass, no controller that
  stores activations and transposes weights. The goal enters through the nudge and nowhere
  else.
- **Every update is local and provably so.** A seam reads two activations; an owner reads
  one. A reference engine settles the net one owner at a time with a message ledger, and
  `conformance` certifies that a fast backend computed nothing an owner could not see.
- **It is a gradient.** With symmetric seams the settlement descends an energy, and the
  local contrast is the loss gradient (equilibrium propagation). The tests check it against
  finite differences.
- **Measured, not claimed.** Every example selects on a validation split, reads its test set
  once, trains the obvious backprop baseline on the same split, and writes a receipt that
  binds every number to the code and data that produced it.
- **Runs where you are.** NumPy float64 for receipts; torch on CUDA or Apple silicon for
  scale, with a dense transport for small nets and a scatter for connectome-sized ones.

## How it learns

<p align="center">
  <img src="https://raw.githubusercontent.com/muellerberndt/cadence/main/docs/assets/learning-cycle.svg" alt="Settle free to an equilibrium; tilt the energy with a nudge on the outputs and settle again both ways; every seam moves on the difference of its own two endpoints" width="100%">
</p>

1. **Settle free.** Clamp the inputs and let every owner repair its own patch until nothing
   moves. The output owners at rest are the answer; no target has entered.
2. **Tilt, and settle again.** Add a small drive on the output owners toward the target
   (`+β`) and, from the same rest state, away from it (`−β`). The net finds a new
   equilibrium each time, and the change reaches the hidden owners through the very seams
   the answer used.
3. **Contrast.** Each seam moves by `η (s⁺ᵢ s⁺ⱼ − s⁻ᵢ s⁻ⱼ) / 2β`, each bias by
   `η_b (s⁺ᵢ − s⁻ᵢ) / 2β`. For a small nudge that is minus the loss gradient.

Labels, a teacher's moves, and rewards all enter the same way: as the target of the nudge
(and, for a reward, its weight). [How it learns](docs/learning.md) has every equation and
a worked six-owner example with every number; [differences](docs/differences.md) sets it
against a feed-forward network with backprop.

## Install

```bash
pip install cadence-net            # NumPy only; the import is `cadence`
pip install "cadence-net[accel]"   # adds torch for CUDA and Apple silicon
```

Python 3.11 or newer.

## Sixty seconds

**A wiring** is `n` owners plus directed overlaps with a contact count and a sign. Build one
from edge lists (a connectome), or let `layered` build a learnable one.

```python
w = cd.Wiring.from_edges(3, pre=[0, 0, 1], post=[1, 2, 2], count=[80, 20, 80], sign=[1, 1, -1],
                         sets={"input": [0], "output": [2]})
```

**A rule** is what an owner does with its inbox. `GradedRule` is the connectome rule;
`learning_rule()` is the one a net that learns needs; `Adaptation` adds rhythm.

```python
engine = cd.Settlement(w, cd.GradedRule(gain=0.02), backend="torch")   # or "cpu"
state = engine.settle(w.members("input"), steps=100, trajectory=True)
state.activation, state.trajectory.shape
```

**A protocol** declares held-out facts with preconditions, and a shuffled control that
keeps every count, sign, and set.

```python
protocol = cd.Protocol(stimuli={"rest": (), "drive": ("input",)},
                       training=[("drive", "output", "active")],
                       rows=[cd.Row("R1", "rest", "output", "inactive", "nothing in, nothing out")])
protocol.score(engine)["passed"], protocol.score(cd.Settlement(cd.shuffled(w, 0), engine.rule))["passed"]
```

**Certify and record.**

```python
cd.conformance(engine, w.members("input"))["max_abs_deviation"]           # ~1e-16 on cpu
receipt = cd.Receipt.build("my-lane/v1", {"score": protocol.score(engine)}, sources=[("lane.py", Path("lane.py"))])
receipt.write(Path("receipt.json")); cd.Receipt.verify(Path("receipt.json"), sources=[("lane.py", Path("lane.py"))])
```

The [quickstart](docs/quickstart.md) does all of this on a connectome, end to end.

## Examples

Everything in [cadence-examples](https://github.com/muellerberndt/cadence-examples) is a
tutorial, a script, a receipt, and for the games a page in which the net settles live. The
[hub](https://claude.ai/code/artifact/ee7a8b53-be8c-4c34-9f91-43d6eaf77be8) links them all;
[How a patch net learns](https://github.com/muellerberndt/cadence-examples/blob/main/HOW_IT_LEARNS.md)
is the tutorial they build on.

| rung | what | receipt says |
|---|---|---|
| [01 digits](https://github.com/muellerberndt/cadence-examples/tree/main/01_digits) | classification, 8×8 digits | 0.962 ± 0.003 held-out in 20 epochs; a same-size MLP: 0.967 in 50 |
| [02 images](https://github.com/muellerberndt/cadence-examples/tree/main/02_images) | MNIST on the accelerator, read out in float64 | 0.9744 in 10 epochs; MLP 0.9779; the two backends agree on every prediction |
| [03 Connect Four](https://github.com/muellerberndt/cadence-examples/tree/main/03_connect_four) | imitate a depth-4 search, then play in the browser | agrees with the search on 0.527 of positions, the MLP on 0.533; both beat random, both lose to depth 2 |
| [04 Pong](https://github.com/muellerberndt/cadence-examples/tree/main/04_pong) | a paddle learns from pixels and reward | see its tutorial: how a reward becomes a nudge, and what credit assignment does to a paddle |
| [05 text](https://github.com/muellerberndt/cadence-examples/tree/main/05_text) | next character of Shakespeare from a sixteen-character window | 3.33 bits per character; bigram 3.71, same-window MLP 3.11, one-layer transformer 3.08 |
| [06 sign writer](https://github.com/muellerberndt/cadence-examples/tree/main/06_sign) | sees a sign, writes it with a two-joint arm | 0.988 overlap with the sign on held-out signs, the teacher's own score |
| [07 chorales](https://github.com/muellerberndt/cadence-examples/tree/main/07_music) | continues Bach chorales chord by chord, with sound in the page | pitch-set F1 0.483, 16.6 bits per chord; same-size MLP 0.470 and 24.7 |
| [08 cart-pole](https://github.com/muellerberndt/cadence-examples/tree/main/08_cartpole) | the classic control task from reward | 154 steps of 500; backprop REINFORCE 392 |
| [09 C. elegans](https://github.com/muellerberndt/cadence-examples/tree/main/09_celegans) | the published connectome under a protocol | fan-in convention 5.2 of 17 held-out ablations vs 1.5 shuffled: a structural signal, not a behavioural model |
| [10 composer](https://github.com/muellerberndt/cadence-examples/tree/main/10_composer) | a living net with feelings: ten days of listening to a piano library, ten of composing from silence by imagining and feeling each phrase; a life, not a training run | 3.4 ideas per context, kernels exactly transposition-equivariant, moods audible in what it plays; silent on 27% of beats and no valence trend yet: three of nine flags fail, and the receipt says why |
| 11 embodiment | a nervous system in a physical body | next |

## What is in the box

| module | gives you |
|---|---|
| `Wiring` | owners and overlaps as sorted arrays, named sets, digests; built from edge lists or by `layered` |
| `GradedRule`, `Adaptation`, `learning_rule` | the owner rule: a graded potential with a rectified sigmoid that emits nothing at rest, an optional leak, and an optional slow variable that turns fixed points into rhythm |
| `Settlement`, `Nudge` | the batched engine on NumPy or torch, with a convergence tolerance and a nudge toward a target; `dense()` for pages |
| `Learner`, `LearnerConfig` | the free/nudged rule: two phases, one local contrast, tied seams, labels or advantage-weighted actions |
| `conformance`, `settle_owner_by_owner`, `Ledger` | the owner-by-owner reference with a message ledger, to certify any backend |
| `blocks` | the block transport: the overlap matrix as dense blocks between the wiring's owner ranges, a still range's product reused, so a step costs what the moving owners cost |
| `timing` | the latency of a decision at the median and the tails, the scheduler's context switches, and the machine's state for a receipt |
| `Protocol`, `Row`, `shuffled`, `select_gain` | declared held-out facts with preconditions, the shuffled control, gain selection under a sparsity cap |
| `Receipt`, `Source`, `fetch` | canonical JSON bound to code and data by digest; pinned public data, downloaded once, verified always |

## Documentation

| | |
|---|---|
| [concepts](docs/concepts.md) | what a patch net is, and why the library is shaped as it is |
| [quickstart](docs/quickstart.md) | from a wiring to a verified receipt in seven calls |
| [learning](docs/learning.md) | the rule in full: every equation, a worked example, every knob |
| [differences](docs/differences.md) | patch net versus feed-forward network with backprop |
| [games](docs/games.md) | imitating a search; learning from reward; setting up credit |
| [tasks](docs/tasks.md) | recipes for every kind of task the ladder and the Kaggle set have met |
| [pages](docs/pages.md) | a trained net settling live in a browser |
| [embodied](docs/embodied.md) | deploying in a body: the loop, several learners in one net, checkpoints |
| [reward](docs/reward.md) | learning from reward: three factors, dreams, what the gates measured |
| [life](docs/life.md) | fast and slow strengths, sleep, pruning and sprouting |
| [protocols](docs/protocols.md) | predicates, the shuffled control, gain selection |
| [brain](docs/brain.md) | every feature of a brain and its simplest constituent here; the language thesis |
| [backends](docs/backends.md) | CPU and torch, precision, the block transport, timing a decision |
| [receipts](docs/receipts.md) | what a verified result is |
| [api](docs/api.md) | every public class and function |

## Against backprop, plainly

Same shape, same count of numbers, same data: on every rung of the examples the rule
reaches the accuracy of the backprop baseline in fewer passes over the data, and on Pong it
learns more from the same rollouts. It costs ten to a hundred times the wall-clock on a
laptop core, because a settlement is tens of steps where a pass is one. It gives no
parameter advantage: a seam is a weight. Every receipt records all three numbers.

## Discipline

1. **Owner-local or nothing.** The reference engine reads one owner and its inbox at a
   time and ledgers every delivery; `conformance` compares any backend against it.
2. **Held out means held out.** Selection on training data only; a test set read once; a
   control that must fail.
3. **A result is a receipt.** Canonical JSON, a digest, the digests of the code and data,
   every pass flag recomputable. A receipt that fails to verify is not a result.

## Where it comes from

Cadence consolidates the lanes of the observer patch net programme: a *C. elegans*
connectome scored against classical ablation phenotypes, the FlyWire *Drosophila* brain
and the MANC nerve cord joined by their descending neurons and scored against held-out
taste, grooming, escape, olfaction, and motor facts, and that nervous system driving a
biomechanical fly in MuJoCo. Every one of those lanes is a wiring, a rule, a protocol, a
control, and a receipt; the library is what they had in common. The learning rule is
equilibrium propagation (Scellier and Bengio, 2017) written for the graded settlement,
with a leak, tied seams, and a centered nudge.

## Status

0.6.0: the core (wiring, rule, engine, reference, protocol, receipts, custody), the
free/nudged learning rule with labels, teachers, and rewards, the three-factor and dream
learners, seams with a life, the block transport and the fused kernel, on NumPy and torch. Nine worked rungs with receipts live in
cadence-examples. On the roadmap: closure sub-nets for in-browser settlement of large
wirings, environment adapters for embodiment, and connectome loaders. Issues and pull
requests are welcome.

MIT licensed.
