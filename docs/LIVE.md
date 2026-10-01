# Use a learned brain with an actual body

Install **0.60.0** as shown in the [quickstart](QUICKSTART.md). This guide
covers live observations, actions, outcomes and continued learning using the
existing public API. Begin with a skill that passes free assessment;
[bootstrapping](BOOTSTRAP.md) explains how to acquire and check one.

A flat, ordinary deep or recursive layout has the same body interface: named
observations go in, qualified outputs come out, and actual consequences supply
experience. Every population currently uses the same processing-patch rule and
participates in one coupled solve. System 1 and System 2 describe routine and
additional corrective roles; adding an observer does not automatically allocate
attention, create independent population clocks, or establish retained correction.
See [brain design](BRAIN_DESIGN.md#measure-speed-and-retained-correction)
for that capability boundary.

## Predict first, execute, then learn from the outcome

This small software body moves according to its position and command. We first
teach a predictor from actual evaluations of that body, then assess it on new
inputs with the future output free. The body equation supplies training data;
it is not wired into the brain.

```python
from math import tanh
from cadence import Cortex, bootstrap

def body(position, command):
    return tanh(0.7 * position + 0.3 * command)

layout = Cortex(seed=2)
senses = layout.input("senses", shape=2)  # current position, actual command
prediction = layout.column("prediction", patches=1, inputs=senses)
layout.output("next_position", shape=1, reads=prediction)
brain = layout.build()

examples = [
    ({"senses": [x, u]}, {"next_position": [body(x, u)]})
    for x in (-0.5, 0.0, 0.5) for u in (-0.5, 0.0, 0.5)
]
checks = [
    ({"senses": [x, u]}, {"next_position": [body(x, u)]})
    for x, u in ((-0.3, 0.2), (0.3, -0.2))
]
report = bootstrap(
    brain, examples, checks=checks,
    epochs=30, batch_size=9, max_error=0.08, seed=2,
)
assert report["passed"], report
for x, u in ((0.2, -0.3), (-0.2, 0.3)):
    assert abs(brain.predict({"senses": [x, u]})["next_position"][0] - body(x, u)) < 0.08

position = 0.25
records = []
for command in (0.4, 0.0, -0.4):
    inputs = {"senses": [position, command]}
    proposal = brain.step(inputs)
    assert proposal["accepted"], proposal["reason"]
    issued_forecast = proposal["outputs"]["next_position"][0]
    next_position = body(position, command)  # the body actually executes now
    records.append((command, issued_forecast, next_position))
    update = brain.observe(inputs, {"next_position": [next_position]})
    assert update["accepted"], update["reason"]
    position = next_position
```

The commands above are supplied; this acquires a body predictor, not an
autonomous control policy. The recorded forecast precedes execution and learning.
A later mismatch is `actual - issued_forecast`. Current internal patch errors
instead compare current state with current prediction, which can change during
repair. They are not an immutable history of earlier forecast mistakes.

In a real adapter, record what the actuator actually did. A movement can be
blocked, or a requested note may have ended. Command flags alone are not physical
feedback. Keep an issued forecast, its context, the executed command and the
measured consequence together. A qualified equilibrium can still predict badly,
and an accurate prediction of failure still calls for a better action.

## Know what is retained

| Kind of state | Owner and meaning |
| --- | --- |
| Live activity | `step` retains qualified patch states; a later call requalifies them |
| Learned relation | Accepted `observe` / `observe_batch` repairs weights and biases |
| Recent observations | An explicit `History` window, supplied as input |
| Pending action and reward replay | A `Reinforcement` learner and the body's execution records |

These mechanisms do not automatically provide episodic memory or protection
against forgetting. `observe` retains its solved activity; `observe_batch`,
including a one-row batch, preserves live activity while retaining parameters.
Pure `settle` and `predict` leave both unchanged.

For measured boundaries, `step(inputs, targets=...)` or `interventions=...`
can retain activity conditioned on known values without learning. These clamps
apply only to that call. A desired future clamp expresses an intention, not a
prediction or witness. Use a separate future-free query before execution and
keep its result distinct from the intended state. Every call still qualifies
the entire connected graph under its current boundaries.

This two-patch example holds a measured past value while the future state stays
free. Output `targets` select exposed coordinates; `interventions` select a
population's coordinates. Both use the same conditional solve:

```python
import json

boundary = Cortex(seed=17)
sensor = boundary.input("sensor", shape=1)
past = boundary.column("past", patches=1, inputs=sensor)
future = boundary.column("future", patches=1, inputs=past)
boundary.output("actual", shape=1, reads=past)
boundary.output("forecast", shape=1, reads=future)
conditioned = boundary.build()
sample = {"sensor": [0.25]}

before = conditioned.snapshot()
query = conditioned.settle(sample, targets={"actual": [0.4]})
assert query["qualified"] and conditioned.snapshot() == before
continued = conditioned.step(sample, targets={"actual": [0.4]})
assert continued == {**query, "accepted": True}
expected = json.loads(before)
expected["state"] = list(continued["state"])
assert json.loads(conditioned.snapshot()) == expected  # only activity changed

held = conditioned.step(sample, interventions={"past": [0.4]})
assert held["accepted"] and held["outputs"]["actual"] == (0.4,)
released = conditioned.step(sample)  # previous clamps are not carried forward
assert released["accepted"] and conditioned.inspect()["admissions"] == 0
```

This demonstrates state handling, not an acquired forecaster: the example has
not learned any body relation. Numerical qualification applies to the free
coordinates under the supplied clamps. A clamped value itself is no prediction.

## Supply recent context explicitly

```python
from cadence import History

history = History(2, steps=3)  # position and executed command at each moment
history.push([0.25, 0.4])
history.push([0.20, 0.0])
encoded = history.push([0.14, -0.4])
assert len(encoded) == history.shape[0]
```

A brain using this encoding declares an input with `shape=history.shape` and
must learn from examples with that same encoding. Each frame contains the
values and a presence mask, oldest to newest. Padding has mask zero; a real
zero-valued frame has mask one. Object visibility, when relevant, is a separate
measured input. `preview` constructs the next encoding without consuming a frame;
`reset` clears the window at an episode boundary.

This is external bounded memory. Once a cue leaves the window, the buffer cannot
recover it. Retained patch activity or extra observer width alone does not prove
learned temporal memory. See the [temporal example](#qualify-temporal-context-and-delayed-credit)
for an acquired recall check and its explicit-history control.

## Learn a discrete choice from reward

Use `Reinforcement` when an executed action receives a reward instead of a
correct-output witness. The helper records transitions, constructs estimated
Q targets and admits them through the same patch rule. It does not turn a
reward directly into a motor target or provide automatic brain-wide emotion.

For a small action set, expose one scalar output per action. Here a supplied
body moves left or right, and moving toward zero earns positive reward:

```python
from cadence import Reinforcement

choices = Cortex(seed=2)
position_sensor = choices.input("position", shape=1)
values = choices.column("values", patches=2, inputs=position_sensor)
for i, name in enumerate(("left", "right")):
    choices.output(name, shape=(), reads=values, indices=(i,))
learner = Reinforcement(
    choices.build(), actions=2, action_input=None,
    value_output=("left", "right"), seed=2,
)

position = 0.6
decision = learner.act({"position": [position]})
assert decision["accepted"]
executed_action = decision["action"]
next_position = position + (-0.1, 0.1)[executed_action]
reward = abs(position) - abs(next_position)
feedback = learner.feedback(
    reward, {"position": [next_position]},
    decision_id=decision["decision_id"], executed_action=executed_action,
)
assert feedback["stored"] and feedback["accepted"]
```

This single transition demonstrates ownership, not acquired navigation. Continue
actual practice and assess later free decisions and complete episodes. The
repository's `examples/live_learning.py` provides bounded learning and reversal
checks. [The reference](REFERENCE.md#reinforcement-discrete-reward-driven-choices) also describes the
alternative scalar value output conditioned on a one-hot action input.

Vector mode uses one value query and a separate `step` to retain activity.
Action-conditioned mode queries each action before that step. All candidate
queries count as work. Exploration and comparison of completed action values
are helper orchestration; all processing and observer states still co-settle.
The selected action's Q output is clamped during fitting; other outputs remain
free and can still contribute to the energy.

## Preserve execution and feedback ownership

Call `act` only when no action is pending. Give the body its `decision_id`, then
pass that ID and the action actually executed to `feedback`. An actuator
override must report the executed action. If a proposed command is discarded,
use `reset()` to abandon it without inventing a transition. Reset does not clear
parameters, activity or replay; construct a new learner for a fresh life.

Invalid feedback leaves the pending decision intact. The first valid feedback
stores the actual transition and consumes the pending action even when fitting
refuses or raises afterward. `stored` therefore does not mean learning succeeded.
Use `replay()` to attempt more learning from the retained experience. An identical
retry of the latest acknowledgment records and learns nothing twice, even when
a newer decision is pending; changed or older acknowledgments are rejected.

Use `terminal=True` only when the reward horizon actually ends; terminal feedback
has no next inputs. A collection pause or command timeout need not be terminal.
For delayed rewards, retain every intervening executed transition with its actual
reward, often zero. Replay can propagate later values backward; this is bounded
Q-learning, not an unlimited-delay guarantee.

The one-step target uses reward `r`, discount `g`, reward scale `R` and value
scale `S`:

```text
y = (1-g) * S * r/R + g * clip(max_a Q(next_context, a), -S, S)
```

Terminal transitions have no future term. With `discount=0`, the helper also
skips future-value queries; nonterminal feedback still requires and retains the
actual next observation. A higher discount reduces the immediate target's
magnitude under this normalization. Longer `credit_horizon` settings have the
explicit greedy-cut semantics in the reference; approximation and exploration
still limit delayed credit.

All replay targets are computed from the pre-update brain, then fitted with
`source="estimate"`. Reward and sensing are facts; the fitted action value is
an estimate. Calling `act(..., explore=False)` disables epsilon exploration,
not learning; exact value ties remain randomized. To freeze parameters, use
`feedback(..., learn=False)` and omit replay. This still changes stored experience,
activity and RNG. Use a `Reinforcement.from_snapshot` copy for isolated assessment.
Nothing schedules background replay automatically.

## Decode discrete actions from settled scores

Action decoding belongs to the application contract. Independent button
scores, preferences over legal compound actions, and estimated action values
have different meanings. Imbalanced button targets can make a zero threshold
miss rare actions, but per-button calibration can improve recall while making
complete actions worse. Compare thresholds or mean-relative decoding with
their controls on separate calibration data, then check legal joint actions
and actual behavior on held-out episodes. For mutually exclusive choices,
argmax over declared action scores is another adapter choice, not a core rule.
Learning can change score distributions, so keep the decoder with the model
and recheck the complete policy after updates. Ranking, agreement and pressed
recall are diagnostics, not substitutes for performed skill.

## A bootstrap, then a life

One live-loop pattern starts with demonstrations, checks acquisition, then
continues learning from consequences. Readiness checks must exercise the
**deployed action path**: agreement from an imitation output does not qualify
a separate action-value output used after takeover. Check subsequent unclamped
decisions on withheld inputs or before admitting each new target, and confirm
behavior in the environment. A teacher's hidden clock or privileged state may
make its choices unlearnable from the supplied sensors. Reaching a training
allowance is a stopping condition, not a readiness pass.

For discrete Q-learning, `Reinforcement` can record outcomes with
`feedback(..., learn=False)` and fit them in separate budgeted `replay()` calls.
Simulator-assisted practice can instead derive targets for `observe_batch`
and train a separate candidate. Both use the common repair law; applications
own target construction and scheduling. Keep one serial owner per brain.
If physics continues while the learner works, bound how long commands remain
valid and record their actual duration, rewards and episode boundaries.
Repeating a command changes the executed transition; rendering FPS does not
measure decision frequency. See [responsive control](#keep-the-body-responsive).

## Watch the equilibrium, not a relaxation film

The public solve returns a final proposal and diagnostics, not intermediate
patch states for a relaxation animation. One call can require many repair
sweeps or refuse at its budget. `settle` is a pure query: repeating it from
unchanged continuation does not advance a partial solve. Use `step` to solve
and retain qualified activity; refused activity is never retained.

Qualification measures constrained stationarity. Prediction errors can remain
nonzero at a qualified equilibrium, and neither quantity measures task skill.
Show changes between qualified endpoint states as inputs change, refusals at
the deployment budget, and complete solve work and latency. Fewer admission
sweeps can reflect conditioning, initialization or batch composition; they do
not demonstrate understanding. Even an admitted repair that lowers its clamped
objective can worsen later free predictions on the teaching inputs. Measure
unclamped prediction quality and executed behavior separately.

## Retain skills and predict the body

`parameter_prior` limits parameter movement within one admission. It is not
protected consolidation, a biological decay constant, or a guarantee that old
skills survive new experience. Batch replay mixes old and new experiences under
the same repair law; always measure old-skill recall after learning the new one.
The reinforcement store is bounded FIFO, so experiences eventually leave it.
Its latest transition is always included; the remaining batch is sampled without
replacement. `replay()` can run between decisions, with a declared work budget.

When continued learning must preserve an existing skill, use a frozen actor
and a separately owned candidate restored from its snapshot. Check both old
skills and new behavior before an explicit replacement, and retain the actor
for rollback. Better mean return can coexist with lost successful episodes.
Repeated gates are development selection; reserve unused episodes for final
confirmation. This orchestration belongs in the application and requires no
new patch primitive. Direct online learning should likewise be compared with
a frozen continuation on matched environment cases.

Count replay's actual coverage of tasks and actions, not only an old/new ratio.
Old demonstrations need not preserve the improved actor's present behavior.
Rehearsing frozen own-score targets uses `source="estimate"` and is not an
exact no-op: joint repair can still move parameters. Test retained behavior.
Batch size and update frequency change the anchor schedule as described in
[batch experience](REFERENCE.md#batch-experience).

For simulator-derived targets, keep measured outcomes separate from the
transformation into preferences. A hard rank discards return magnitude, so
shrinking a reward bonus need not shrink a teaching update. Bind cached action
comparisons to the actual context, action duration, horizon and frozen
continuation that produced them. They answer “this action, then that policy”;
a repaired policy needs its own behavioral evaluation. Fit these derived
targets with `source="estimate"`; hypothetical brain predictions alone are
not new environment witnesses.

Action-conditioned consequence learning already uses witnessed targets:

```text
inputs  = previous sensory history + actual executed action
targets = subsequently measured position, contact or other body consequence
brain.observe(inputs, targets)  # actual measurements, not imagined outcomes
```

Design those input/output ports explicitly. Teaching a body predictor does not
by itself teach which action to prefer. Conversely, estimating reward does not
learn an accurate visual model of the body. The `live_control.py` example learns
a small body relation, queries actions, and chooses by predicted need reduction;
that comparison is an application controller, explicitly separate from neural
settlement. `live_learning.py` exercises reward-based choice and reversal.

An executed motor command is a factual record, but using it as a motor teaching
target trains imitation of that choice regardless of its consequence. Repeating
this with the brain's own choices can overwrite a useful routine. Preserve the
action in the causal record, distinguish consequence prediction from preference
learning, and check retained behavior after online updates. Connected populations
alone do not guarantee that outcome quality changes the motor parameters.

## Drives and curiosity

Keep physical state in the body adapter: energy, fatigue, contact and actuator
strain are measured quantities. Supply relevant needs as sensors, and define
their consequences as bounded rewards. For example, food increases an energy
reserve, motion consumes it, and a reward can measure reduced energy deficit.
That supplies a preference; it does not script a route or select the food.

```python
from cadence import LearningProgress

curiosity = LearningProgress(rate=0.1)
assert curiosity.update("body_prediction", 0.5) == 0.0
bonus = curiosity.update("body_prediction", 0.2)
assert 0.0 < bonus <= 1.0
```

Supply prediction error against a **subsequently observed outcome**, measured
before teaching that outcome. Do not use numerical stationarity as curiosity.
The helper tracks an exponentially smoothed error by a bounded context key and
returns positive relative error reduction. This is a progress heuristic; noisy
decreases can also earn a bonus. It is neither information gain nor a reliable
noise detector. Check it against constant-error, random-noise and bonus-disabled
controls before relying on it for exploration. Context categories and mixing
weights are declared adapter choices or candidate genes, not learned facts.

A possible bounded composition is
`reward = (1-curiosity_weight)*external_reward + curiosity_weight*bonus`
with external reward in [-1,1] and weight in [0,1]. The application owns this
choice. Built-in homeostatic chemistry, learned neuromodulation, dynamic
plasticity gating and consolidation are not supplied. `exploration`, replay
frequency and `parameter_prior` are explicit functional controls; do not relabel
them as measured dopamine or claim they reproduce biological physiology.

## Keep the body responsive

This controller tries three commands using the learned body predictor above.
The host chooses the smallest predicted distance from zero. This is explicit
candidate search, not a learned planner or internal attention mechanism; all
three whole-brain queries count as decision work.

```python
from time import monotonic, sleep
from cadence import LiveController, slew

def decide(observation):
    candidates = []
    for command in (-0.4, 0.0, 0.4):
        result = brain.settle({"senses": [observation["position"], command]})
        if not result["qualified"]:
            return {"qualified": False, "command": (0.0,)}
        distance = abs(result["outputs"]["next_position"][0])
        candidates.append((distance, command))
    return {"qualified": True, "command": (min(candidates)[1],)}

# Check the callback before handing its brain to the worker.
check = decide({"position": 0.25})
assert check["qualified"] and check["command"] == (-0.4,)
controller = LiveController(decide, fallback=(0.0,), max_age=0.25)
try:
    controller.submit({"position": 0.25})
    # A rendering/physics tick reads immediately, using fallback if necessary.
    command = controller.read()
    actuator = slew((0.0,), command, rate=2.0, dt=1/60)
    # Demo verification only: wait outside the rendering loop for completion.
    deadline = monotonic() + 2.0
    while controller.inspect()["completed"] == 0 and monotonic() < deadline:
        sleep(0.001)
    metrics = controller.inspect()
    assert metrics["qualified"] == 1 and metrics["errors"] == 0, metrics
finally:
    exited = controller.close(timeout=1.0)
assert exited  # only now may another owner access this brain
```

One worker owns the callback and all brain calls it performs, including learning.
Do not access that brain from another thread. One pending sensory sample may be
replaced by a newer sample; never put the only copy of an unprocessed reward or
executed action into that lossy slot. Keep ordered experience in the serial owner
or a separate lossless application queue. A control decision that was never
executed must not become an action/reward transition.
When a callback uses `Reinforcement.act`, the body adapter must acknowledge
execution before supplying `feedback`. If the command expired or was discarded,
call `reset` from that same serial owner to abandon the pending action. The
generic worker does not infer whether a physical command was executed.

Command age starts at sensory submission, including queue time. Refused,
erroneous, expired or closed results use the supplied fallback. `slew` limits
actuator change, leaving target selection to the controller. Physics, animation
and low-level support remain identifiable. This thread wrapper is best effort:
Python's GIL and OS scheduling affect latency, GPU startup takes time, and an
in-flight solve is not cancelled. A false result from `close` means the worker
still owns its brain. Sweep budgets are not wall-clock deadlines. Measure both
decision latency tails and fresh-command rate, separately from rendering FPS.

## Save a whole life and run the small gates

`Reinforcement.snapshot()` includes the brain, replay records, pending action and
random generator. Restore it with `Reinforcement.from_snapshot`. History and
curiosity have their own validated snapshots; save them at the same paused
boundary as the learner and the environment. A brain-only snapshot has no
external sensory history, replay dataset or body state.

A complete application save should include:

| Owner | What to save |
| --- | --- |
| `Reinforcement` | Its snapshot, including the brain, replay, pending action and RNG |
| `History` and `LearningProgress` | Their separate snapshots, if used |
| Body/environment | Physical state, episode position, observation encoding, environment RNG and simulation clock |
| Execution adapter | Command/outcome IDs, whether the pending action was executed, and any unprocessed consequences |

Pause collection and drain or persist outcome messages under one serial owner
before taking these snapshots. A learner's pending action records a choice; it
does not certify execution. On restore, deliver saved consequences only for
an action actually executed; abandon a discarded command with `reset`. Never
re-execute an already acknowledged action. `LiveController` has no snapshot:
stop its worker and check that `close()` returned true before accessing its
brain elsewhere, then create a new controller for the resumed owner.

Brain checkpoints and reinforcement checkpoints bind their implementation
sources. An execution override exists on `Brain.from_snapshot`, but not on
`Reinforcement.from_snapshot`; rebuilding only the brain does not transfer a
whole learner's replay and pending-action state. Keep the original compatible
package version for saved lives.

Run the bounded examples from the repository:

```sh
python examples/live_learning.py
python examples/live_control.py
```

These are small capability checks, not a complete pet or proof that recursion
beats a flat model. The tests cover hidden-cue recall through explicit history,
delayed reward and reversal, replay retention, saved continuation, consequence
prediction, numerical refusal and stale-command handling. Longer memory,
continuous-action reinforcement, learned planning and a browser creature remain
separate demonstrations.

## Qualify temporal context and delayed credit

`examples/temporal_credit.py` separates two bounded capabilities:

- **Recall from supplied history.** Opposite cues precede identical distractor
  suffixes and final observations. Delays are 2, 4 and 8 ticks; `History` contains
  `delay + 1` frames. Reserved tests use new cue amplitudes and distractors.
  Flat, ordinary-connected and observing layouts each have four patches, with
  different contacts and parameter counts. Retained activity, reset activity
  and removed-history controls distinguish explicit context from native memory.
- **Delayed reward.** Two actions are available; only the final transition
  rewards the first choice. Sensors disclose the current stage and first
  executed action, giving sufficient state to isolate credit assignment.
  Delays 0/2/4/8 mean 1/3/5/9 actual transitions. One-step TD with replay is
  compared with discount-zero learning and frozen parameters. No preferred
  choice or desired Q value enters the sensors.

Each reward life collects 60 episodes, then evaluates 40 with learning disabled
and exploration still 0.4. The per-case executed-choice gate is 0.65; an optimal
policy under that exploration has expected success 0.8. Discount 0.8 attenuates
remote reward, and shared function approximation can change even unvisited
values. This is not an unrestricted-delay guarantee or a memory-capacity test.

Run the fixed seeds 2 and 7 (seed 0 is the development/CI fixture):

```sh
python examples/temporal_credit.py --out /tmp/temporal-credit.json
```

The JSON retains every scheduled case, source hash, configuration, outcome,
saved/resumed comparison, refusal, work count and execution timing. Resuming a
pending executed action does not execute it twice; a collection pause does not
invent a terminal transition. These are separate controlled demonstrations,
not one integrated autonomous life or evidence of an observer advantage.

The [stored receipt](../examples/receipts/temporal_credit.json) is **historical
0.50.0 qualification evidence**, produced at commit `d9b592c` before that version
bump. Within that release transition, only the package version string changed
among its hashed implementation files. It is not a 0.60.0 rerun. Its 66
cases include 18 passing memory cases and 16 passing TD cases; the other cases
are controls. Current reruns must retain their own source hashes, results and
timings rather than inheriting those numerical conclusions.
