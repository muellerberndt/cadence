# Evaluate a brain

New to evaluation? [Evaluate without fooling yourself](teach.md#evaluate-without-fooling-yourself)
covers the basics: held-out data, enough samples, seeds, per-class accuracy and a control. This
page is the full method for a capability claim.

This method applies to any task, demo or capability claim. Test one acquired
brain through bootstrap, routine use, disturbance, repair and saved
continuation ([the continuing life](world-model.md#bootstrap-use-repair-continue));
isolated classification is a mechanism control. A uniform-random policy with
the same observations and actions is the baseline for every claim: a brain that
does not beat it has learned nothing worth reporting. Declare a change's scope
as [CONTRIBUTING.md](../CONTRIBUTING.md#scope-and-evidence-for-behavior-over-time) describes.

## Start at the boundary

Give every external port a meaning, a unit, an availability condition and a
source. An absent observation and an observed zero are different readings: give
a feature that can be absent a missing flag, in witnessed and imagined readings
alike. Keep goals, executed actions, measured consequences and private
predictions apart even when their arrays share a shape; the application
supplies these meanings.

Supply correct actions to the environment first and check what happens, then
compare the learner's actions through the same boundary: this separates a
learning failure from an interface that cannot realize the behaviour. An
action named after a sensor reading (brightness as pitch, onset count as
retriggers) is a supplied instrument design that needs its own test.

The [Doom body](../benchmarks/doom/README.md) is a worked boundary: the rendered screen, three
declared own-body ports and a history port go to the player; the scorer reads the engine's other
variables and credits each outcome once; completion, death, deadline and truncation stay
distinct; a refused action leaves the world at its tic; and a checkpoint of game, body and brain
resumes the same life at the same tic, the brain to the bit on `backend="cpu"`.

## Check which differences survive encoding

Encode two cases whose difference matters. If both map to the same ports, no
deterministic learner recovers the difference from those ports; context or a
prior can support a guess without making the observation unique.

A counterexample: one event per grid interval at times `j + 0.25`, against
`j + 0.25` for even `j` and `j + 0.75` for odd `j`. The first is evenly spaced,
the second alternates long and short gaps. Both have the same event count,
occupied intervals and unsigned distance to the nearest grid point, so an
unsigned timing-spread summary cannot tell them apart. Expose signed offsets
when the task depends on them.

## Baselines and controls

| Control | What it rules out |
| --- | --- |
| Uniform-random policy | Any claim of learned behaviour; required for every claim |
| Majority answer or training-set mean | Gains from the output distribution alone |
| Last-observation persistence | Gains from slow change in the world |
| Fitted linear predictor | Gains a linear map over the same inputs reaches |
| Tabular learner | Gains from memorising discrete states |
| Online recurrent actor-critic ([`benchmarks/competitors.py`](../benchmarks/competitors.py)) | Gains any recurrent learner with the same information reaches |
| Shuffled labels, cues or wiring ([the shuffled control](../benchmarks/instruments/README.md#the-shuffled-control)) | Gains that do not depend on the claimed relation |
| One memory switched off ([check that a memory does the work](memory.md#check-that-a-memory-does-the-work)) | Credit given to the wrong memory |

Give every arm the same information, replay policy, update count and compute
budget (the [key-door benchmark](../benchmarks/keydoor/README.md) runs random,
tabular and recurrent arms beside the brain). A copied target, a new decoder,
an added intention input or a larger action vocabulary makes a different task.

For memory claims, the [coupled-memory experiment](../benchmarks/attention/README.md)
separates joint-equation and local-credit checks from learned relevance, and the
[two-binding protocol](../benchmarks/attention/HEADS_PROTOCOL.md) tests independent heads
against width, port and actual-work controls and separates contextual association-state
transfer from a context-free observation code. Causal relevance, semantic addressing,
retention and continuing-life integration each need their own evidence.

## Report per-behaviour errors

Common easy outputs hide errors on rare decisive ones, so report the error per
behaviour (timing, intention, the rare event) beside the aggregate. Test
perception, memory, transitions and decisions separately: novel combinations,
delayed cues, effects of alternative actions and rollout error by horizon.

Where probabilities are claimed, report log loss and calibration of a valid
normalized distribution beside accuracy.

Test ambiguity before adding capacity: swapped intention, cue removal, context
removal and activity reset. When tasks learn alone but average under joint
training, inspect cue activity, output sensitivity, competing gradients,
horizon and capacity. Changing what one moment stands for (event,
frame, bar) changes the task; count path length, ports, state and compute.

## Test coherent behavior over time

Sequential output, sensory feedback, the working trace, the associative stores
and efference are all legitimate. Next-event accuracy and self-feeding success
alone are component results. For an integrated sequence claim, declare the
context and time span, then compare the same acquired brain with retained,
absent and perturbed context under matched exposure. Test recovery after a
disturbance and counterfactual contexts that need different continuations, and
keep the simpler control, earlier skills, refusals and work in the gates.

Separate equation residual from meaningful surprise and from task success.
`Brain.act` qualifies the present state with trace and memory as held input and
settles no joint future trajectory. Supplied clocks, phase or goals belong in
the declared information budget. The [world-model
guide](world-model.md#coherent-behavior-over-time) gives the brain's side.

## Close the action and consequence loop

For an acting system, test the whole sequence:

1. Observe the situation and retain the relevant context.
2. Form a private action proposal under a sustained intention.
3. Predict its consequences with relations learned from actual action and
   consequence records.
4. Execute the selected action and measure the consequence.
5. Repair the discrepancy, then retest earlier skills.

`Brain.imagine` settles private branches and changes no live state. A planner
that detunes a prediction toward a goal can approach a goal no action can cause:
replay the chosen action with the goal removed, execute it and measure the
outcome. Demonstrations omit most mistakes: add the learner's own interactions
and test transfer on whole held-out tasks.

## Keep branches isolated and provenance explicit

Planning starts from the interpretation of actual evidence, and a preference
must never remove an observed obstacle or rewrite an earlier action. For a
reproducible decision, freeze parameters, live context, record contents
and address statistics, preprocessing, random state and any external planner
memory. Compare live state, counters and normalization statistics before and
after every private computation. Keep observed outcomes, teaching targets and
imagined outcomes distinct; one executed action is evidence for that action
only. Keep both the proposal and the executed command when an actuator clips.
Future observations are unavailable in a rollout unless an evaluation condition
grants them: predict every later state the transition consumes, or declare an
internal body model. Changing a key encoder or the model can make stored
records inconsistent, so freeze, version or rebuild keys with it. The [three
clocks](settling.md#three-clocks) assign each update to its operation.

## Count the full work

Charge sensing, validation, memory reads and writes, free and teaching phases,
eligibility, replay, imagination and refused attempts. The brain reports its
share in `brain.last_settlement` (free-answer sweeps in `steps`) and
`brain.learner.updates`. With arousal enabled, `live` adds `brain.last_arousal`
(`sweeps`, `learning_sweeps`) and the per-mode totals `brain.arousal.moments`
and `brain.arousal.sweeps`. `record_settlements` sees every trajectory,
learning and imagination included.

Give bootstrap, routine and recovery cost and their total over the life, with
attempted and committed updates and phase failures. Routine answers should
keep outcome quality on held-out situations at lower work and respond to
changed conditions. Keep every scheduled seed, including unattempted and
time-capped cells, compare equal checkpoints only and add operation counts to
wall-clock time. A residual and the model energy measure the equations, never
joules. [Receipts](#the-cost-fields) hold the cost fields.

## Separate the causes of improvement

| Cause | Controlled comparison |
| --- | --- |
| Unique experience | Nested sets of distinct observations at equal update count and batch size |
| Training exposure | The same set and initialization, read at predeclared update counts |
| Capacity | The same observations, ports, objective and evaluation at different widths |
| Temporal scope | Longer experienced sequences, counting episodes and teaching values |
| Internal readback | Aligned feedback against disconnected, shuffled and matched recurrent controls |
| Consolidation | Direct replay of the same retained evidence |

Keep evaluation data separate, use several initializations, report each paired
result, and show acquisition, retention and task quality at small scale before
a long run. Equal update counts or widths alone do not match resources.

## Comparing with other model families

For a transformer, an MLP or a recurrent network as the comparison:

1. **Declare the complete state and equations,** with every observing, observed
   and decision population. An observer takes part in the equations; an
   external callback does not. Previous-event boundaries can stay fixed;
   current-event populations settle together.
2. **Qualify the whole solution.** Report the maximum equation defect over every
   state component, per-level residuals, budgets and nonfinite outcomes. Failed
   cases stay in the denominator with their work charged.
3. **Name the training rule.** `Learner` contrasts free and nudged phases
   locally, qualified under `LearnerConfig(qualified=True)`. A finite unroll
   with a backward scan and an implicit equilibrium gradient are separate arms,
   each with its own check.
4. **Test causal participation.** Perturb base and observer populations both
   ways, cut feedback, freeze the observer's readback and compare with a
   sequential observer; only target-free activity is carried.
5. **Match the experiment.** Same information, labels, masks, splits, history
   and pinned revisions; selection apart from held-out evaluation; parameter-
   and compute-matched arms with comparable tuning and seeds. Count settling,
   adjoint work, retries, memory and selection, inference apart from training;
   name device, precision and implementation for timings.

## Coordination, self-readback and creativity

A coordinating patch, a deeper observer or a self-readback port earns its place
through a named behaviour it improves against the internal-readback controls
above, with goals and observations in every arm, added state counted, and
depth, iteration, addressing and supervision varied separately. Linear feedback
can be absorbed into recurrent weights, so extra levels alone show nothing. For
specialization, perturb one region and measure its own contribution.

Creativity is a novel response that serves a purpose or meets constraints. Ask
whether it differs substantially from the acquired examples, meets the
requirements, improves through revision and stays useful in a changed context;
random variation, recall and interpolation do not qualify. Count search,
supplied structure and external evaluators, and keep human feedback verbatim
beside the objective checks. Judge the goal too: compare the target, the
executed result and an unchanged reference. Feedback that rejects target and
result together points to goal selection.

## Common missteps

| Misstep | Why it fails | Check |
| --- | --- | --- |
| Independent classification presented as a continuing world model | Fresh state omits the context being claimed | Test [one acquired brain](start.md#one-continuing-equilibrium-brain) across bootstrap, use, disruption and repair |
| A small residual taken for a useful model | The equations can hold for a wrong interpretation | Measure task quality, retention, routine work and repair cost |
| The answer produced by a trained external readout | The extra model carries the capability | Read the brain's declared output; report the readout as a baseline |
| Free-state energy read as surprise | The causal recurrence has zero defect for a wrong prediction too | Compare the prediction with the actual reading |
| More replay called more data | Replay adds exposure without new experience | Vary unique examples, update count and capacity separately |
| Predicted state after an action called feedback | The measured consequence never entered | Declare which actual readings enter the next call |
| Afterglow taken for short-term memory | Persistent activity can lose the distinction | Remove the cue, vary delay and context, reset activity |
| Protected responses counted as useful memory | Protection keeps an error as faithfully as a skill | Check usefulness, later correction and remaining learning capacity together |
| Imagined outcomes fed back as observations | Predictions reinforce their own mistakes | Keep readings, proposals, teaching and imagination distinct |
| Scale expected to restore discarded information | Many sequences share one summary | [Test encoding collisions](#check-which-differences-survive-encoding) |
| A named module added before locating a failure | It hides an encoding, objective or optimization problem | Compare correct supplied actions, each task alone, joint learning and controls |
| Supplied routing called learned specialization | Assigned coordinate groups impose roles | Compare fixed connectivity against dense and disconnected controls |
| Content sampled from scores where the model predicts at chance | The scores carry no information there: one draw is noise and consecutive draws are chaos | Measure held-out accuracy at those positions separately; let the model decide when something happens and declare what may happen |
| A measurement's prior tuned until the data show the expected structure | The score rises because the prior wrote the labels, and the labelled events lose their relation to the source | Audit the fixture before training: faithfulness to the source against plain labelling, and the events of interest above chance where they should be |

## Independent-sample controls

`predict(observations)`, `accuracy(observations, labels)` and `fit(observations,
labels, *, epochs=30, batch=32)` treat rows as independent samples: qualified cold
solves that read no trace or associative memory. `fit` starts every stream afresh, keeps
the arousal state as it is and returns training accuracy per epoch; a refused score keeps
accepted lessons. Use them as controls on a separate instance; greedy `act` on a loaded copy
measures the memory-aware response.

```python
import numpy as np
from cadence import Brain

cues = np.eye(4)  # four one-hot observations

control = Brain.compose(4, 2, modules=(16,), seed=1)
control.fit(cues, np.array([0, 1, 0, 1]), epochs=2)
assert control.predict(cues).shape == (4,) and 0 <= control.accuracy(cues, [0, 1, 0, 1]) <= 1
```

## Receipts

A receipt stores outcomes, an embedded digest and a manifest of selected source files.
The caller decides which outcomes to record and what the verifier checks.

### Write and verify a small receipt

Save this as `receipt_demo.py` and run `python receipt_demo.py`. It binds a small
arithmetic result to its own script and to the installed receipt implementation:

```python
from pathlib import Path
import cadence as cd

sources = [
    ("receipt_demo.py", Path(__file__)),
    ("cadence/receipts.py", Path(cd.__file__).with_name("receipts.py")),
]
values = [1, 2, 3]
receipt = cd.Receipt.build(
    "sum-demo/v1", {"values": values, "total": sum(values)}, sources=sources,
)
path = receipt.write(Path("sum_receipt.json"))

def check(body):
    if body["values"] != [1, 2, 3] or body["total"] != sum(body["values"]):
        return "stored sum differs from the declared inputs"
    return None

ok, message = cd.Receipt.verify(path, sources=sources, check=check)
assert ok, message
assert cd.Receipt.read(path).digest == receipt.digest
print(message)
```

Expected output: `canonical form, digest, sources, arithmetic agree`. For an experiment,
bind every consumed source and dataset, and supply checks for its schedule, outcomes
and metrics.

### Shape

```json
{
  "kind": "experiment/v1",
  "body": { "...": "recorded readings, rows, gain tables, custody" },
  "source": { "files": [{"path": "experiment.py", "sha256": "..."}], "manifest_sha256": "..." },
  "digest": "sha256 of the canonical JSON of kind, body and source"
}
```

`Receipt.build(kind, body, sources=())` hashes each `(relative, path)` pair in `sources`
(an empty tuple gives an empty manifest) and the canonical JSON of the payload
(`cd.canonical_json`: sorted keys, no whitespace). Mapping keys become strings, tuples become
lists, NumPy arrays and scalars and `Path` values become plain JSON, and NaN or infinity
raises `ValueError`. `write(path)` creates parent directories, writes the newline-terminated
canonical form and returns the path; `Receipt.read(path)` loads a file without checking it;
`to_dict()` returns the four fields. `cadence.receipts.canonical_sha256(value)` and
`cadence.receipts.source_manifest(pairs)` (not exported at top level) build the digest and the
`source` block.

### Verification

`Receipt.verify(path, *, sources=None, check=None)` returns `(ok, message)` and stops at the
first disagreement: (1) the file is byte for byte the canonical form of its content;
(2) the embedded digest matches; (3) when `sources` is supplied, the complete source
manifest matches; (4) when `check` is supplied, `check(body)` returns no problem text (for a
protocol receipt: every pass flag follows from the stored readings and every tally from the
rows). The message lists what was checked, or names the disagreement or the check's text.
This verifies the declared source inventory; it cannot discover omitted dependencies.
The attention and depth runners bind every library Python module, package metadata, the
provenance helper and their protocol scripts, and record the effective numerical mode,
dependency versions and thread environment.

A receipt registered in `benchmarks/historical-receipts.json` verifies against its archived
source bytes, at the evidence scope it was recorded with, and replays no training:

```sh
python benchmarks/provenance.py --historical benchmarks/attention/results/held-heads.json
python benchmarks/provenance.py --historical benchmarks/depth/results/and-count.json
```

Whole-bank head maxima, distinct relevant facts and answer accuracy are separate
measurements; two globally different slots can belong to the same fact.

### What belongs in the body

The connectome summary and digest (`Connectome.summary()`, `Connectome.digest()`) with the
custody block for measured data; the graph description (`NeuralGraph.to_dict()`: connectome
summary, neuron model, backend, transport, layout); the gain selection table with every gain
tried and its admissibility; the protocol as data, with the reference for every row; the
score with readings and reference readings on every row, and the control's score; the
backend's conformance report (`cd.conformance(graph, stimulus)` on a `NeuralGraph`); and a
boundary block saying what the experiment supplies rather than derives and what it does not
claim.

### The cost fields

Every arm of a comparison carries the same accounting over the life: for a composed brain
the free-answer sweeps of `last_settlement`, `learner.updates` and, under `live`, the
per-mode totals `arousal.moments` and `arousal.sweeps`
([count the full work](#count-the-full-work)). An efficiency comparison also
reports elapsed time, peak memory, learning and search work and every failed attempt. Test
targets never choose a baseline; record the disjoint calibration and validation split and
its seeds.

### Editing invalidates, on purpose

A source check fails once any file in the manifest changes. Record source hashes beside the
library version, because one version string can cover several source revisions. A receipt
is evidence for its own source snapshot: keep its bytes and verify against that snapshot
instead of re-signing an old result against later code. A run records data generation,
seeds, every scheduled condition, predictions and targets, and its verifier rejects missing
outcomes instead of averaging whichever rows are left.
