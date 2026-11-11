# Instruments: protocols and timing

Two benchmark instruments of a source checkout, not part of the installed package:
`protocol.py` scores a brain against held-out facts with preconditions, and `timing.py` times
a decision. Put this folder on `sys.path` to import them (`import protocol`, `import timing`).
Their tests, `test_*.py` here, run in the default `pytest` collection. Every name below
comes from the module it is introduced with: `from protocol import Protocol, Row, Levels,
evaluate_predicate, shared_code, shuffled, select_gain, PREDICATES`.

## Protocols: held-out tests with preconditions

A `Protocol` is data: stimuli, the few training facts a brain may be shown, held-out rows and
threshold levels. Use it to score a brain built from a connectome against facts from the
literature and to choose a gain from the training facts alone.

### Stimuli, rows, training facts

```python
import cadence as cd
from protocol import Levels, Protocol, Row

connectome = cd.Connectome.from_synapses(
    100, pre=[0, 1], post=[2, 3], count=[200, 200],
    populations={"sensors": [0, 1], "motors": [2, 3]},     # the other 96 neurons stay at rest
)
neuron_model = cd.NeuronModel(gain=0.02)
brain = cd.NeuralGraph(connectome, neuron_model)

protocol = Protocol(
    stimuli={"rest": (), "touch": ("sensors",)},     # name -> populations driven at 1.0
    training=[("touch", "motors", "active")],         # (stimulus, readout, predicate[, versus])
    rows=[
        Row("R1", "rest", "motors", "inactive", "no input, no output"),
        Row("R2", "touch", "motors", "reduced", "cutting the sensors", ablate=("sensors",)),
    ],
    levels=Levels(active=0.5, inactive=0.2, margin=0.15),
    steps=60,
)
report = protocol.score(brain)
print(report["passed"], "of", report["total"], "rows;", report["training_passed"], "training facts")
```

`Row(id, stimulus, readout, predicate, reference="", ablate=(), relative_to="",
tier="experiment", versus="")` is one held-out fact: `reference` is a citation string, `ablate`
lists populations zeroed for the row, `tier` groups results. The comparison reading comes from
`versus` when set, otherwise from `relative_to`: a stimulus name is read on the intact brain, a
population name under the row's own stimulus and ablation (for `exceeds` and `lateralized` a
population wins over a stimulus of the same name), and an empty name means the row's stimulus on
the intact brain. An unknown nonempty name raises `KeyError`.

`score(brain)` settles each needed (stimulus, ablation) pair once for `steps` steps and returns
`training`, `training_passed`, `rows`, `passed`, `total` and `passed_by_tier`. Each result holds
the row's or fact's identifying fields, its `reading`, `passed` and, for rows, the comparison
reading as `reference`. A reading is `{mean, fraction}` over the readout, `fraction` counted at
`code_level`. `neurons_for(connectome, stimulus)` lists a stimulus's neurons; `to_dict()` returns
the protocol, its levels and the `PREDICATES` text as JSON-ready data.

`Levels` holds the thresholds, with defaults `active=0.5`, `inactive=0.2`,
`margin=0.15`, `sparse_min=0.005`, `sparse_max=0.2`, `densify_margin=0.05`, `specific_max=0.25`
(the share of their union two codes may have in common) and `code_level=0.5` (the activation at
or above which a neuron counts in `fraction` and in a code). The table below shows which
predicate reads which level.

### Predicates and their preconditions

| predicate | passes when | precondition |
|---|---|---|
| `active` | mean(readout) ≥ `active` | |
| `inactive` | mean(readout) ≤ `inactive` | |
| `reduced` | mean(readout) ≤ mean(reference) − `margin` | mean(reference) ≥ `active` |
| `retained` | mean(readout) ≥ `active` | mean(reference) ≥ `active` |
| `released` | mean(readout) ≥ mean(reference) + `margin` | mean(reference) ≤ `inactive` |
| `exceeds` | mean(readout) ≥ `active` and ≥ mean(other) + `margin` | |
| `lateralized` | absolute mean difference ≥ `margin` | the larger mean ≥ `inactive` |
| `sparse` | `sparse_min` ≤ fraction(readout) ≤ `sparse_max` | |
| `densified` | fraction(readout) ≥ fraction(reference) + `densify_margin` | the reference is sparse |
| `specific` | the readout's codes under the row's stimulus and under `versus` share at most `specific_max` of their union | both fractions ≥ `sparse_min` |

A failed precondition fails the row, so a silent brain cannot pass `reduced` and a saturated
one cannot pass `released`. A `specific` row names its second stimulus,
`Row("F1", "fruit", "kc", "specific", "fruit and yeast are told apart", versus="yeast")`, and a
training fact does the same with a fourth element, `("fruit", "kc", "specific", "yeast")`.
Select a population's gain on this fact before learning there
([brains from a connectome](../../docs/anatomy.md#brains-from-a-connectome)): a code that is the same for every stimulus gives a
lesson nothing to attach to.

`evaluate_predicate(predicate, value, reference, levels=None)` is the pure test on two
readings, so stored readings can be rechecked without the brain; `None` means `Levels()` and an
unknown predicate raises `KeyError`. For `specific`, `value["shared"]` comes from
`shared_code(activation, other, members, level)`. `PREDICATES`
maps each predicate to its rule as text.

### The shuffled control

`shuffled(connectome, seed, *, keep=None)` permutes postsynaptic endpoints and keeps every
count, sign, out-degree and named population, with no autapses. `keep` is a boolean mask of
synapses left in place, such as the bridges between two joined datasets; a mask of the wrong
length, or a graph that cannot avoid autapses, raises `ValueError`. Score the same protocol on
several shuffle seeds with the same neuron model and gain: a pass that fails on the shuffles
shows that the particular connections matter under the declared model and protocol, and only
that.

### Gain selection under a sparsity cap

```python
from protocol import select_gain

make_brain = lambda g: cd.NeuralGraph(connectome, neuron_model.replace(gain=g))
gain, table = select_gain(make_brain, protocol, grid=(0.01, 0.02, 0.03, 0.05), sparsity_cap=0.05)
print(gain, [(t["gain"], t["facts_passed"], t["admissible"]) for t in table])
```

`select_gain(make_brain, protocol, grid, *, sparsity_cap=0.05)` scores every candidate on the
training facts only. A candidate is admissible when at most `sparsity_cap` of all neurons sit at
or above 0.5 under every training and `versus` stimulus. The winner passes the most facts among
admissible candidates, smallest value on ties, and may fail some or all of them; each table
entry has `gain`, `facts_passed`, `fraction_active`, `admissible` and `readings`.
`make_brain` may vary any number, such as one population's `log_gain` in
`cd.NeuralGraph(connectome, model, log_gain=...)`. An empty grid or no admissible candidate raises
`ValueError`; declare a wider grid or another cap before evaluation. `sparsity_cap=None` admits
toy brains allowed to become fully active. Keep the whole table with the result, for example in
a [receipt](../../docs/evaluate.md#receipts).

### For learned brains

`Protocol` scores settled responses to fixed stimuli. A brain that learns from experience is
evaluated with the same habits: select on a validation split of the training data only, read
the test set once after selection, and include a control that should fail, such as a brain
trained on shuffled labels scoring at chance
([baselines and controls](../../docs/evaluate.md#baselines-and-controls)).

## Timing a decision

`timing.latency(decide, repeats=1000, warmup=20)` runs `decide()` `warmup` times, then
times `repeats` calls (an integer of at least 1; `warmup` at least 0) and returns `repeats`,
`p50_us`, `p90_us`, `p99_us`, `max_us`, `mean_us`, `jitter` (`(p99 - p50) / p50`),
`voluntary_switches` and `involuntary_switches` from `getrusage`, and `environment`.
`timing.environment()` records machine, system, cores, Python version, BLAS thread
limits, CPU affinity on Linux, load average and the NumPy, Numba and torch versions. On
Windows the load and switch counts are `None`.

```python
from timing import latency

graph = cd.NeuralGraph(connectome, neuron_model)
inputs = [{int(i): 1.0} for i in connectome.populations["sensors"]]
life = {"t": 0, "state": None}
def decide():  # a new input each call, continuing from the previous state
    life["t"] += 1
    life["state"] = graph.settle(inputs[life["t"] % len(inputs)], steps=30, state=life["state"])
    return life["state"].activation
report = latency(decide, repeats=50, warmup=5)
print(round(report["p50_us"]), report["environment"]["cores"])
```

`decide()` must advance a changing input and carry or reset state as the deployment would
(one unchanged warm input times a stable-state fast path) and synchronize accelerator work.
Report cold and warm results, output quality and residuals with the time.
