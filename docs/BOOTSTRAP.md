# Teach a skill before relying on it

Start with the [quickstart](QUICKSTART.md) for 0.60.0 installation and a complete
first learning loop. This guide explains how to prepare a useful teaching stream,
check acquisition and continue learning without confusing a qualified solve
with a competent brain.

**Bootstrapping** is initial guided practice. **Live operation** uses the same
brain in its environment and may include more learning. These lifecycle phases
are separate from System 1 and System 2: an ordinary deep skill may become a
routine, while a recursive observer may help correct it. Flat, ordinary deep
and observing layouts all use the same public learning operations and processing
patch rule. See [brain design](BRAIN_DESIGN.md) for the architecture and its
current limits.

## A small complete acquisition check

This supplied calibration relation is `answer = 0.6 * signal`. The four teaching
values, two development checks and two final assessment values are distinct.
The final assessment contains no output clamps.

```python
from cadence import Brain, Cortex, bootstrap

layout = Cortex(seed=2)
signal = layout.input("signal", shape=1)
response = layout.column("response", patches=1, inputs=signal)
layout.output("answer", shape=1, reads=response)
brain = layout.build()

examples = [
    ({"signal": [x]}, {"answer": [0.6 * x]})
    for x in (-0.8, -0.4, 0.4, 0.8)
]
checks = [
    ({"signal": [x]}, {"answer": [0.6 * x]}) for x in (-0.6, 0.6)
]
report = bootstrap(
    brain, examples, checks=checks,
    epochs=20, batch_size=4, max_error=0.1, seed=2,
)
assert report["passed"], report
assert report["accepted"] == 4 * report["updates"]
for x in (-0.5, 0.5):
    prediction = brain.predict({"signal": [x]})["answer"][0]
    assert abs(prediction - 0.6 * x) < 0.1

saved = brain.snapshot()
restored = Brain.from_snapshot(saved)
assert restored.predict({"signal": [0.5]}) == brain.predict({"signal": [0.5]})
```

`bootstrap` validates and copies the complete dataset before changing the
brain. It checks free recall and development accuracy before learning and after
each completed epoch. It stops when both meet `max_error`, the epoch allowance
runs out, or a solve refuses. Earlier admitted updates survive a later refusal.
`epochs=0` checks readiness without teaching.

`max_error` uses your output units. Solver `tolerance` concerns stationarity of
the common energy and does not certify accuracy. The checks influence stopping,
so they are development data; reserve other cases for assessment. A passing
report applies to those cases, not to arbitrary environments.

## Decide what each teaching row means

| Available evidence | Teaching operation | What it can acquire |
| --- | --- | --- |
| Measured sensor or body consequence | `observe` / `observe_batch` with its default `source="witness"` | Prediction of that consequence |
| An actual demonstration action | The same witness operations | Imitation of that action from the supplied context |
| A transformed target, prediction or simulator-derived preference | The same operations with `source="estimate"` | The explicitly declared estimated relation |
| Reward following an executed discrete action | `Reinforcement` action/feedback/replay | Estimated action values; see [live learning](LIVE.md) |

A reward is not a desired motor value. Teaching a body predictor does not itself
teach which command to prefer. Repeatedly teaching the brain its own executed
command trains imitation, even if the consequence was poor. Keep actual outcomes
and the transformation into teaching targets distinct. Source labels bind retry
identity; they do not authenticate the caller's evidence.

Supply enough causal information to distinguish the answers. Position alone
may omit velocity. A demonstrator using a private clock cannot be imitated
reliably from observations that exclude it. Preserve actual actuator feedback
where commands are ambiguous: a held musical note may be silent because its
sample ended. Relevant voice age or measured sound is a body observation,
not an instruction to activate a population.

A bounded `History` can supply recent observations and actual actions. It is
explicit input memory, not learned recurrent memory. Batch rows have private
states and acquire no chronological relationship merely by being adjacent.

## Choose connected capacity, then measure it

| Relation to learn | Small starting layout |
| --- | --- |
| One direct sensor-to-answer relation | One directly sensing output patch |
| Several direct relations | One selected patch per output coordinate |
| Learned nonlinear combinations | A small ordinary representation, for example four patches, feeding an output population |
| Useful internal mismatch readback | An observer over that representation, compared with the ordinary control |

These are starting constructions, not validated recipes for raw vision, music
or general planning. Width only helps an output if the added patches are
connected to it. Increasing a flat population from one to ten patches while
reading only its first state does not create a ten-patch hidden representation.
Ordinary deep populations already return influence through the coupled energy;
observers additionally read exact current prediction errors.

Inspect wiring before training:

```python
description = brain.inspect()
assert description["output_connected_patches"] == 1
assert description["outputs"][0]["sensor_coverage_by_coordinate"] == (1,)
```

`fan_in=None` connects every declared source coordinate. Sparse `fan_in` is an
explicit information/cost choice; aggregate sensor coverage does not guarantee
that each selected output uses all sensors. The inspection fields describe
structural paths, not measured causal influence: zero weights and saturation
can still suppress a path.

The [layout examples](VARIANTS.md) teach and resume flat, ordinary deep and
recursive brains. Try the same interface first; add depth when a controlled
comparison improves free task performance enough to justify the work. Equal
patch counts need not imply equal parameters, connections or cost. A useful
routine may be deep; an observer alone does not demonstrate System 2.

## Keep units and sampling consistent

Choose an input transform and output encoding before training. With default
bounds, targets comfortably inside `[-1, 1]`, such as `[-0.6, 0.6]`, are a
practical small-example choice. Fit scaling or whitening only on teaching data
and preserve that preprocessing with the model. The library does not infer it.

Increasing `state_bound` permits larger clamps but does not expand the `tanh`
prediction range. For a free output patch that no other patch reads or observes,
`state = tanh(drive) / (1 + state_prior)` unless a bound is active. A target of
`2` can qualify when clamped with `state_bound=4` and still be impossible to
recall. Decode normalized predictions back into application units.

For mutually exclusive classes, use a score per class and an explicit decoder,
such as argmax over legal actions. Scores are not normalized probabilities.
When exposing several scalar outputs from one population, give each a different
`indices` selection; otherwise they all default to its first patch.

Sampling changes what the brain learns. A predictor taught mostly on betting
events cannot interpret their frequency as the frequency of bets across all
opportunities. Keep factual forecasts representative; attach selected-action
values only where that action and consequence were recorded. Missing outcomes
are not zero rewards. Leaving an output unclamped removes its target, not its
patch from the joint energy: the free patch can still affect parameter repair.

Count distinct events separately from replayed presentations. Rebalancing rare
actions is a supplied curriculum choice. Evaluate both rare decisions and
ordinary behavior afterward; a high average score can hide failure to learn
the maneuver the task needs.

## Individual and batch admissions

The bootstrap helper's default `batch_size=1` uses ordered `observe` calls,
which retain activity as well as parameters. A larger batch uses
`observe_batch`: each row has private activity, while parameters are shared.
The objective is mean example energy plus one pre-batch parameter anchor.
A successful batch retains parameters and one event identity, preserving live
activity. A refused batch retains none of that batch's proposal.

```python
live_state = brain.state
update = brain.observe_batch(examples, event_id=100)
assert update["accepted"]
assert len(update["states"]) == len(examples)
assert brain.state == live_state

# Retry the exact acknowledged event without teaching it twice.
retry = brain.observe_batch(examples, event_id=100)
assert retry["duplicate"] and not retry["accepted"]
```

Use explicit, increasing event IDs for external experience that may be retried.
The identical latest retry must keep the method, ordered rows, physical clamps
and source label. Older or changed identities are rejected. Automatic IDs
advance on acceptance, but do not identify an external retry. See
[batch experience](REFERENCE.md#batch-experience) for the full contract.

| Counter | Meaning in `bootstrap` |
| --- | --- |
| `presentations` | Attempted example rows, including refused groups |
| `accepted` | Example rows in admitted calls |
| `updates` | Accepted atomic learning calls |
| `work` | Counted numerical work, including assessment queries |

Four admitted rows can mean one update. Changing batch size changes the number
of parameter anchors per epoch and the learning trajectory. A final short
batch has its own mean and anchor. Equal presentation counts do not imply equal
work or equivalent learning. Even a one-row `observe_batch` preserves activity,
which differs from `observe`.

Batching can improve throughput, but does not supply delayed credit or memory.
For large recordings, load bounded row lists into `observe_batch`; bootstrap's
`batch_size` limits each joint solve, not its up-front dataset allocation.
Save data position, sampling RNG and preprocessing alongside the brain.

## Continue learning without losing the earlier skill

`parameter_prior` anchors each admission to the parameters at its start. A
larger value resists movement; it is not zero-centered weight decay or protected
long-term consolidation. Smaller or larger priors can help particular datasets,
so compare them on fixed development data and keep failures and search work.
Likewise, more initialization scale can supply nonlinear curvature while making
settlement harder. There is no task-independent setting that ensures learning.

Interleave relevant earlier examples with new experience and check both skills
without targets after updates. A newly accepted fit can improve its clamped
objective while worsening free predictions or behavior. Accurate prediction of
a poor outcome is still poor behavior. When testing a new live-learning rule,
keep a frozen continuation and a routine-plus-learning control to distinguish
learning benefit from a benefit that specifically required recursive planning.

For higher-cost deployments, a separate candidate restored from the actor's
snapshot can be assessed before replacement. This is optional application
orchestration, not an extra learning rule or automatic brain-wide attention.
Keep serial ownership of each brain and preserve its original actor for rollback.
Repeated development checks select a candidate; they are not independent final
confirmation.

## Check the actual closed loop

Before scaling up, test the deployed output and decoder in the environment with
the teacher disconnected. Agreement on demonstration states is insufficient:
one wrong action changes which states the learner visits. If practice collects
new demonstrations on those states, label it as a supervised curriculum and
compare it with an equal-exposure replay control.

For a control task, measure routine behavior, an actual disturbance, recovery,
and retained competence after further learning. Record the original forecast
before each consequence. Test both unexpected outcomes and predictable failure
to meet the goal. No hidden phase label or special attention input should make
the test answer available to the actor. See [the live guide](LIVE.md).

A useful assessment records:

- Free prediction error, decoded actions and actual task outcomes.
- Earlier-skill retention, rare decisions and individual seeds as well as means.
- Refused/capped calls, accepted updates, distinct events and replay exposure.
- All query, learning and assessment work; complete action latency separately.

## Diagnose before adding compute

| Observation | Check next |
| --- | --- |
| A solve refuses | `reason`, `stationarity`, `work`, bounds and budget, from the unchanged continuation |
| Solves qualify but teaching examples cannot be recalled freely | Input sufficiency, output connectivity, units, representation and sampling |
| Recall passes but new cases fail | Coverage, generalization, decoder and the contexts encountered by the policy |
| A new skill displaces an old one | Replay coverage, update frequency, parameter anchoring and explicit retention checks |
| Observers add work without benefit | A capable ordinary control, equal factual information and total task cost |

The default budget allows up to 2,048 accepted repair sweeps; rejected proposals
and qualification also cost work. Changing `tolerance` changes the admission
criterion, not the amount of skill acquired. Do not loosen it merely to turn a
refusal into a claimed success. More exposure cannot repair absent information
or a disconnected representation.

## Scale execution after acquisition works

Optional CPU/GPU tensor execution preserves the same objective and final
float64 qualification; see [acceleration](ACCELERATION.md). Batch rows and
independent brains can run in parallel. Calls admitting experience to one brain
remain ordered, anchored to its preceding accepted parameters. Do not average
independently learned checkpoints or concurrently mutate a shared brain.

The repository example owns one simulated body and brain per process:

```sh
python examples/parallel_bootstrap.py --lives 4 --workers 2
```

Measure complete acquisition time, memory, refusals and fresh behavior before
claiming a speedup. More hardware supplies neither missing sensory information
nor evidence that recursive observation improves the task.
