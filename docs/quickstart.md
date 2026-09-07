# Quickstart: a measured wiring, a held-out test, a receipt

This walks the whole loop on a wiring you build yourself, then shows the same calls on a
connectome edge list. Everything runs on the NumPy backend; add `backend="torch"` for a GPU.

## 1. Build a wiring

```python
import numpy as np
import cadence as cd

rng = np.random.default_rng(0)
n, e = 200, 1500
w = cd.Wiring.from_edges(
    n,
    pre=rng.integers(0, n, e),
    post=rng.integers(0, n, e),
    count=rng.integers(5, 40, e),                 # contacts per overlap
    sign=np.where(rng.random(e) < 0.3, -1.0, 1.0), # 30% inhibitory
    sets={"sensors": range(0, 10), "motors": range(190, 200)},
    min_count=5,                                   # drop weak overlaps, as connectomes do
)
w.summary()
```

`Wiring` sorts overlaps by (post, pre), merges parallel ones, drops autapses, and digests
itself. Sets are tuples of owner rows and travel with the wiring, so protocols speak in names.

## 2. Settle it

```python
rule = cd.GradedRule(gain=0.02)                 # drive per contact per unit activation
engine = cd.Settlement(w, rule)
state = engine.settle(w.members("sensors"), steps=60, trajectory=True)
state.mean(w.sets["motors"]), state.active(), state.trajectory.shape
```

Rest is an exact fixed point: with no clamp nothing fires. The activation is a sigmoid
re-based to emit zero at rest, so a quiet net stays quiet.

## 3. Declare what you will test

```python
protocol = cd.Protocol(
    stimuli={"rest": (), "touch": ("sensors",)},
    training=[("touch", "motors", "active")],           # the one fact the model may see
    rows=[
        cd.Row("R1", "rest", "motors", "inactive", "no input, no output"),
        cd.Row("R2", "touch", "motors", "reduced", "cutting the sensors", ablate=("sensors",)),
    ],
)
```

Predicates carry preconditions: `reduced` requires the intact net to have been active, so a
dead net cannot pass it.

## 4. Select the gain on the training fact, under a sparsity cap

```python
gain, table = cd.select_gain(
    lambda g: cd.Settlement(w, rule.replace(gain=g)), protocol, grid=(0.01, 0.02, 0.03, 0.05),
)
engine = cd.Settlement(w, rule.replace(gain=gain))
```

A gain is admissible only while at most 5% of owners are active under the training
stimuli; pass `sparsity_cap=None` for toy nets that are meant to light entirely. Above the
cap the net runs away and every readout lights, which says nothing about the wiring. The
table records every gain tried.

## 5. Score, and score the control

```python
report = protocol.score(engine)
control = protocol.score(cd.Settlement(cd.shuffled(w, seed=0), rule.replace(gain=gain)))
report["passed"], control["passed"]
```

`shuffled` permutes postsynaptic endpoints and keeps every count, sign, out-degree, and
named set. If the wiring passes what the control does not, the prediction came from the
wiring.

## 6. Certify the engine

```python
cd.conformance(engine, w.members("sensors"), steps=60)
# {'backend': 'cpu', 'max_abs_deviation': 1e-15, 'ledger': {'clean': True, ...}, ...}
```

The reference engine reads one owner and its inbox slice at a time and counts one delivery
per declared overlap per step. Run this on the torch backend too, and record the number.

## 7. Write the receipt

```python
from pathlib import Path

receipt = cd.Receipt.build(
    "quickstart/v1",
    {
        "wiring": w.summary(),
        "rule": engine.rule.to_dict(),
        "gain_selection": {"selected": gain, "table": table},
        "protocol": protocol.to_dict(),
        "score": report,
        "control": control,
        "conformance": cd.conformance(engine, w.members("sensors")),
    },
    sources=[("quickstart.py", Path(__file__))] if "__file__" in globals() else [],
)
receipt.write(Path("receipt.json"))

def check(body):
    for row in body["score"]["rows"]:
        if row["passed"] != cd.evaluate_predicate(row["predicate"], row["reading"], row["reference"]):
            return f"row {row['id']} pass flag does not follow from its readings"
    return None

cd.Receipt.verify(Path("receipt.json"), check=check)
```

## The same calls on a connectome

A connectome is an edge list with a synapse count and a presynaptic sign. Pin the file,
fetch it once, verify it always, and build the wiring the same way.

```python
src = cd.Source(
    key="edges", file="edges.csv", url="https://example.org/edges.csv",
    sha256="<64 hex chars>", citation="Who measured it, where it was published",
)
paths = cd.fetch([src], root=Path("data"), allow_download=True)
pre, post, count, sign = load_your_csv(paths["edges"])   # your parser
w = cd.Wiring.from_edges(n_neurons, pre=pre, post=post, count=count, sign=sign, min_count=5,
                         sets={"sugar_grn": [...], "mn9": [...]})
```

From there the protocol, the control, the conformance check, and the receipt are the
same six calls, with `cd.manifest([src])` embedded in the receipt body as custody.
