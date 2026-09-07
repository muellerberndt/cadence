# Cadence

**Machine learning by patch-net settlement: owner-local repair, held-out tests, receipts.**

A patch net is a set of *owners*, each holding one patch of state, joined by declared
*overlaps*. Nothing is computed globally. Every owner repairs its own patch from what
arrives over its overlaps, and the state the net comes to rest in is the answer. Cadence
is the library for building, settling, testing, and certifying such nets, from a
six-owner ring to a 161,827-owner nervous system read from a connectome.

```python
import cadence as cd

wiring = cd.Wiring.from_edges(4, pre=[0, 1, 2, 3], post=[1, 2, 3, 0], count=[120] * 4)
engine = cd.Settlement(wiring, cd.GradedRule(gain=0.03))
engine.settle(clamp={0: 1.0}, steps=60).activation.round(2)
# array([1., 1., 1., 1.])
```

## What is in the box

| layer | what it gives you |
|---|---|
| `Wiring` | owners and overlaps as sorted arrays, named sets, digests; built from edge lists |
| `GradedRule`, `Adaptation` | the owner rule: a graded potential with a rectified sigmoid that emits nothing at rest, and an optional adaptation variable that turns fixed points into rhythm |
| `Settlement` | the engine, on NumPy float64 (`"cpu"`) or torch (`"torch"`: CUDA, Apple silicon, or CPU) |
| `conformance`, `settle_owner_by_owner`, `Ledger` | an owner-by-owner reference engine with a message ledger, to certify that a fast backend computes nothing the owners could not |
| `Protocol`, `Row`, `shuffled`, `select_gain` | declared stimuli, readouts, and held-out facts with preconditions; the shuffled-wiring control; gain selection under a sparsity cap |
| `Receipt`, `source_manifest` | canonical JSON bound to code and data by digest, verified by recomputing every pass flag |
| `Source`, `fetch` | pinned public data, downloaded once, verified always |

## Install

```bash
pip install cadence-net            # NumPy only; the import is `cadence`
pip install "cadence-net[accel]"   # adds torch for CUDA and Apple silicon
```

Python 3.11 or newer. On an M-series Mac the torch backend runs on MPS in float32; on CUDA
it runs in float64. The CPU backend is always float64 and is the one receipts are made on.

## Sixty seconds

**A wiring** is `n` owners plus directed overlaps with a contact count and a sign. Build it
from edge lists; parallel overlaps merge, autapses drop, and you can name sets of owners.

```python
w = cd.Wiring.from_edges(
    3, pre=[0, 0, 1], post=[1, 2, 2], count=[80, 20, 80], sign=[1, 1, -1],
    sets={"input": [0], "output": [2]},
)
```

**A rule** says what an owner does with its inbox. `GradedRule` is the one every
connectome lane uses. Add `Adaptation` when you want rhythm.

```python
rule = cd.GradedRule(gain=0.02, adaptation=cd.Adaptation(tau_steps=40, strength=1.0))
```

**Settle** from rest under a clamp. A clamp is a list of owners at full amplitude, a
`{owner: level}` map, or a dense drive vector. Ask for the trajectory when you want to watch.

```python
engine = cd.Settlement(w, rule, backend="torch")   # or "cpu"
state = engine.settle(w.members("input"), steps=100, trajectory=True)
state.activation, state.trajectory.shape
```

**Declare a protocol** and score it. Rows are held-out facts with predicates that carry
their preconditions. The shuffled control keeps every count, sign, and set.

```python
protocol = cd.Protocol(
    stimuli={"rest": (), "drive": ("input",)},
    training=[("drive", "output", "active")],
    rows=[cd.Row("R1", "rest", "output", "inactive", "nothing in, nothing out")],
)
protocol.score(engine)["passed"], protocol.score(cd.Settlement(cd.shuffled(w, 0), rule))["passed"]
```

**Certify** the backend and **write a receipt**.

```python
cd.conformance(engine, w.members("input"))["max_abs_deviation"]
receipt = cd.Receipt.build("my-lane/v1", {"score": protocol.score(engine)}, sources=[("lane.py", Path("lane.py"))])
receipt.write(Path("receipt.json"))
cd.Receipt.verify(Path("receipt.json"), sources=[("lane.py", Path("lane.py"))])
```

The [quickstart](docs/quickstart.md) walks through a connectome; [concepts](docs/concepts.md)
explains why the library is shaped this way; [backends](docs/backends.md) covers devices
and precision; [receipts](docs/receipts.md) covers what a verified result means.

## Discipline

Three rules the library enforces rather than recommends:

1. **Owner-local or nothing.** The reference engine reads one owner and its inbox at a
   time and ledgers every delivery. `conformance` compares any backend against it.
2. **Held out means held out.** A protocol names the few facts a model may be shown. Gains
   are selected on those alone, and only while the net stays sparse, because runaway
   activity lights every readout and proves nothing about the wiring.
3. **A result is a receipt.** Canonical JSON, a digest, the digests of the code and data,
   and every pass flag recomputable from the stored readings. A receipt that fails to
   verify is not a result.

## Where it comes from

Cadence consolidates the lanes of the observer patch net programme: a *C. elegans*
connectome scored against classical ablation phenotypes, the FlyWire *Drosophila* brain
and the MANC nerve cord joined by their descending neurons and scored against held-out
taste, grooming, escape, olfaction, and motor facts, and that nervous system driving a
biomechanical fly in MuJoCo. Every one of those lanes is a wiring, a rule, a protocol, a
control, and a receipt; the library is what they had in common.

## Status

Version 0.1.0 is the core: wiring, rule, engine, reference, protocol, receipts, custody.
On the roadmap: the owner-local free/nudged learning rule, closure sub-nets for
in-browser settlement, environment adapters for embodiment, and connectome loaders.

MIT licensed.
