# Learning through a continuing life

Cadence separates three kinds of memory. **Live activity** is the last admitted
settled state. **Long-term memory** is the retained weights and biases repaired
by experience; their ability to change is plasticity. **Temporal context** is
an explicit record of recent observations. These are different mechanisms.
Neither saved activity nor a larger observer population automatically provides
working memory, episodic retrieval, or protection against forgetting.

The helpers originated in `cadence-net` 0.50.0. This development copy uses the
unreleased candidate's explicit `decision_id` and `executed_action` feedback
arguments; installing released 0.50.0 does not supply that signature. See the
[candidate migration guide](MIGRATION_060.md) before running these examples.
The helpers reuse the same patch equation and qualified admission. They add
no mandatory dependency and do not simulate neurotransmitter chemistry. See
the [reference](REFERENCE.md) for every parameter and failure contract.

## Keep one body interface

A flat, deep ordinary or recursively observing brain receives named observations
and returns named outputs through the same calls. One body adapter executes a
qualified command and reports what happened. Internal observers require no
separate evaluator or attention signal from the application.

Keep three records distinct: the original forecast before execution, the command
actually applied, and the later measured outcome. Feed back actual body state;
a requested motor movement can be blocked and a requested sound can have ended.
Current patch residuals compare current states with current predictions; they
do not store the error of an earlier forecast. Preserve that forecast and its
context before learning from the outcome.

The desired routine → disturbance → correction → inexpensive routine cycle is
an integrated capability still to be established. The helpers below provide
explicit history, action/outcome ownership and learning calls; they do not
implement automatic internal attention or choose the task objective. See
[brain design](BRAIN_DESIGN.md#spend-compute-according-to-measured-need) for
behavioral checks that distinguish successful recovery from mere settlement.

## Remember a recent observation

If a measured body value corresponds to an output or population, `step` can
hold that value during repair and retain the resulting activity without
learning. The same `targets` and `interventions` arguments are available on
pure `settle` queries. For example, this small layout demonstrates continuation
from a supplied past observation; it has not acquired a forecasting skill:

```python
from cadence import Cortex

layout = Cortex(seed=4)
sensor = layout.input("sensor", shape=1)
past = layout.column("past", patches=1, inputs=sensor)
future = layout.column("future", patches=1, inputs=past)
layout.output("actual", shape=1, reads=past)
layout.output("forecast", shape=1, reads=future)
continuation = layout.build()
result = continuation.step({"sensor": [0.1]}, targets={"actual": [0.3]})
assert result["accepted"] and continuation.state[0] == 0.3
assert continuation.inspect()["admissions"] == 0
```

Each later call starts from that activity and requalifies its eligible states.
Resupply any clamps that still apply: `step` does not persist them as rules.
Clamping a future output to a desired goal conditions a solve on an intention;
it does not predict that goal or record a learning witness. Use a separate
future-free solve to evaluate the forecast before execution.

```python
from cadence import Cortex, History

history = History(2, steps=3)  # two sensory values per moment
cortex = Cortex(seed=2)
context = cortex.input("history", shape=history.shape)
base = cortex.column(patches=4, inputs=context)
observer = cortex.observer(patches=2, observes=base)
cortex.output("answer", shape=1, reads=observer)
brain = cortex.build()

history.push([0.8, 1.0])
history.push([0.0, 0.0])
inputs = {"history": history.push([0.0, 0.0])}
result = brain.step(inputs)
assert result["accepted"]
```

Each history block contains the supplied values and a presence mask; blocks run
oldest to newest. Padding has mask zero, so a real zero-valued sample remains
distinguishable. An occluded object needs an application visibility indicator,
as in the second coordinate above. The presence mask indicates an actual frame,
not whether every object was visible in it. `preview` computes the next encoding
without consuming a frame; `reset` clears the window at an episode boundary.

This is bounded **external history**, presented to the jointly settling brain.
The brain must learn how to use it. After a cue leaves the window, this mechanism
cannot recover it. The snippet above only constructs and queries a history-fed
brain; it has not taught recall. See the [temporal qualification](#qualify-temporal-context-and-delayed-credit)
for acquired recall on reserved sequences. Long-lived learned recurrent memory
remains a distinct task.
Do not call adding history alone a demonstrated recursive-memory advantage.

## Learn choices from consequences

`Reinforcement` evaluates discrete actions through a scalar action-value output
of the same brain. The action enters as a one-hot sensor. Every processing and
observer population remains in each coupled solve. Epsilon exploration and the
comparison between completed action queries are explicit orchestration outside
that equilibrium; they are not a new neural readout or an emergent planner.

For a small fixed action set, a more efficient option exposes **one scalar output
per action from a single jointly settling brain**. Set `action_input=None` and
pass those output names as `value_output`. A learning update clamps only the
chosen action's output; the other patches remain free in that same solve. This
avoids a separate query for every action and lets action ranks vary directly
with the current sensory context. Cadence Pet uses this form. `act` still
performs a separate `step` to retain qualified activity: vector mode uses two
solves per successful decision, while action-conditioned mode uses one query
per action plus that `step`.

```python
from cadence import Reinforcement

choices = Cortex(seed=2)
odor = choices.input("odor", shape=2)
values = choices.column(patches=3, inputs=odor)
choices.observer(patches=2, observes=values)
for i, name in enumerate(("rest", "left", "right")):
    choices.output(name, shape=(), reads=values, indices=(i,))
policy = Reinforcement(choices.build(), actions=3, action_input=None,
                       value_output=("rest", "left", "right"))
```

```python
from cadence import Reinforcement

layout = Cortex(seed=2, initial_scale=1.5)
senses = layout.input("senses", shape=3)
action = layout.input("action", shape=2)
perception = layout.column(patches=6, inputs=(senses, action))
reflection = layout.observer(patches=3, observes=perception)
layout.output("value", shape=(), reads=reflection)
learner = Reinforcement(layout.build(), actions=2, seed=2)

decision = learner.act({"senses": [0.3, 0.1, 0.0]})
assert decision["accepted"]
action_index = decision["action"]
# The environment executes action_index and returns its actual consequence.
admission = learner.feedback(
    0.0, {"senses": [0.2, 0.1, 0.0]},
    decision_id=decision["decision_id"], executed_action=action_index,
)
assert admission["stored"]
```

Call `act` only when there is no pending action. Pass its `decision_id` and the
action actually executed to `feedback` with that action's consequence. An
actuator override must report the executed action instead of the proposal.
An identical retry of the latest outcome is acknowledged without recording or
learning twice, including while a newer decision is pending. A conflicting or
older outcome is rejected without changing the brain or pending decision.
`feedback(..., terminal=True)` has no next inputs
and no future-value term. An arbitrary collection timeout is not necessarily
terminal: use the next observation when future rewards continue. If reward arrives
later, intervening transitions can have zero reward; subsequent replay propagates
the later reward backward through learned value predictions. This is one-step
Q-learning with replay, not an eligibility trace or unlimited-delay guarantee.

For reward `r`, discount `g`, reward scale `R` and value scale `S`, the estimated
target is:

```text
y = (1-g) * S * r/R + g * clip(max_a Q(next_context, a), -S, S)
```

The second term is zero at a terminal state. With `discount=0`, replay also
skips future-value queries and uses only the immediate reward; a larger
`credit_horizon` does not add future credit. Nonterminal feedback still needs
the actual next observation. This normalization keeps targets
inside the declared output scale for bounded rewards, rather than silently
truncating accumulated returns. It represents `(1-g)*S/R` times discounted
return. Higher discount reduces immediate target magnitude: it is not a free
increase in horizon. The finite output range and function approximation still
limit accuracy. No convergence or task-independent default guarantee is made.

Replay samples stored transitions, computes every target using the pre-update
brain, then calls `observe_batch(..., source="estimate")`. The numerical patch
rule is unchanged. The source label distinguishes a derived teaching target
from an actual observation, including in retry identity. It does not authenticate
the caller's evidence. Real reward and next sensing are observed; the fitted
action value is an estimate. Ordinary witnessed demonstrations still use
`observe` or `observe_batch` with their default `source="witness"`.

Invalid feedback does not consume the pending action. A first valid
acknowledgment stores the transition and consumes the action even if its learning
attempt refuses. Retry with `replay`, not by pretending the outcome happened
twice. Query/fit refusals do not change learned parameters. Check `accepted`
before applying actions and after learning; `stored` alone does not mean learning
succeeded. `feedback(..., learn=False)` records a frozen-learning control.
The stored record also survives an exception during the subsequent fit; it
remains an actual experience even though no fitted update was admitted.

`feedback` binds the outcome to the issued decision and deduplicates an identical
retry of the latest acknowledgment. A body adapter must retain that decision ID
and report which command it actually executed. Earlier identities are rejected;
Cadence does not retain an unlimited network-message history.
`reset()` abandons a pending action without inventing feedback; it does not clear
replay, parameters or retained activity. Start a new learner for a fresh life.

`explore=False` disables epsilon exploration for a decision; it does not freeze
learning, and exact value ties are still broken randomly. For a frozen-parameter
evaluation, pair executed actions with `feedback(..., learn=False)` and omit
`replay()` calls. Feedback still records transitions and changes the replay
store; queries via `act` still retain activity and advance the helper's RNG.
For a completely isolated assessment, evaluate a `Reinforcement.from_snapshot`
copy. Call `replay()` explicitly if you want extra updates: nothing schedules
background learning for you.

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

```python
from cadence import LiveController, slew

def decide(observation):
    result = brain.settle(observation)
    return {"qualified": result["qualified"],
            "command": result["outputs"]["answer"]}

controller = LiveController(decide, fallback=(0.0,), max_age=0.25)
try:
    controller.submit(inputs)
    # A rendering/physics tick never waits for a completed settlement.
    command = controller.read()
    actuator = slew((0.0,), command, rate=2.0, dt=1/60)
finally:
    exited = controller.close(timeout=1.0)
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

`examples/temporal_credit.py` separates two bounded capabilities. In the cue
fixture, opposite initial cues have identical distractor suffixes and final
observations. Delays are 2, 4 and 8 observation ticks; the supplied `History`
window is `delay + 1`. Training uses cue amplitudes ±0.8 and two distractor
sequences, stopping checks use ±0.4 and a third sequence, and reserved tests
use ±0.3/±0.6 and two new sequences. Flat, ordinary-connected and observing
layouts each contain four patches. Their edges and parameters differ; this
is a capability comparison at matched patch count, not a depth advantage claim.
Each layout is queried with retained or reset activity, and again with history
removed. A saved brain and history resume partway through every test episode.
Retaining activity alone is not assumed to preserve an occluded cue.

The credit fixture has two actions and a supplied one-hot observation of the
current stage and the first executed action. Subsequent actions leave that
choice unchanged. Only the final transition supplies reward, +1 or −1 according
to the first choice. Delays 0/2/4/8 therefore mean 1/3/5/9 executed transitions.
The two possible rewarded choices are tested separately for every seed. Neither
the preferred choice nor a desired value enters the sensors or a witness target.
This fixture provides sufficient observed state to isolate reward credit from
the separate history experiment; it does not establish learned recurrent memory.

Each life collects 60 episodes, then executes 40 evaluation episodes with
learning disabled and exploration still 0.4. Success measures those actual
choices, not just greedy value rankings. The declared behavioral gate is at
least 0.65 success per TD case; an optimal policy with this exploration has
expected success 0.8. Frozen parameters and discount-zero learning are controls,
not additional acquisition claims. Disabling bootstrapping can still change
unvisited-state values through shared parameters, so its measured behavior is
reported rather than assumed to be chance. Discount is 0.8 for TD, replay stores
256 transitions and samples up to eight per update. The ideal first-choice
value magnitude is `0.18 * 0.8**delay`; attenuation and finite approximation
error limit the useful horizon. This tests one-step TD with replay, not eligibility
traces, unrestricted delays or a comparison of every return estimator.

One serial owner pairs each `act` with the transition it executes. Event time
is an integer simulation tick with one discount factor per transition; there is
no variable wall-time discount. At most one action awaits feedback. Terminal
means the reward horizon has actually ended. A collection pause instead saves
and resumes the learner, including a pending executed action, replay and RNG;
it does not create a terminal transition. A refused action is never executed.
A refused learning attempt preserves its actual transition but no parameter
update. Abandoned commands use `reset`; hypothetical `settle` queries own no
pending action and cannot receive feedback.

Run the bounded confirmation on seeds 2 and 7 (seed 0 is the development and
CI fixture):

```sh
python examples/temporal_credit.py --out /tmp/temporal-credit.json
```

The JSON preserves every scheduled case, trial outcome, resumed comparison,
source hash, learning configuration and helper-level solver-work sum, including
candidate queries, replay, evaluation, refused calls and continuation twins.
The report also records complete wall and CPU time. The source hashes must
remain unchanged during the run. These are two controlled demonstrations, not
one integrated autonomous life, learned episodic retrieval or evidence that
observers outperform conventional recurrent models.

The [source-bound confirmation receipt](../examples/receipts/temporal_credit.json)
contains all 66 cases for seeds 2 and 7. It was produced at qualification commit
`d9b592c`, before the 0.50.0 version bump. Of its hashed implementation files,
only the package version string subsequently changed; the receipt retains its
original hashes. Reruns on the release have different version-file hashes and
timings, so do not expect byte-identical report JSON. All 18 memory cases pass, with maximum
reserved full-history error 0.107. All 16 TD cases exceed the 0.65 executed-choice
gate; their individual success rates range from 0.725 to 0.900. The following
means pool both rewarded choices and both seeds (160 evaluation episodes per
cell):

| Reward delay | TD with replay | Discount-zero learning | Frozen parameters |
| --- | ---: | ---: | ---: |
| 0 | 0.750 | 0.750 | 0.500 |
| 2 | 0.806 | 0.625 | 0.500 |
| 4 | 0.794 | 0.363 | 0.500 |
| 8 | 0.781 | 0.631 | 0.500 |

Every saved/resumed comparison matched and no solve refused. The complete run
recorded 291 seconds wall time and 275 seconds CPU time while other local checks
were running; these are execution receipts, not production latency claims.
Controls sometimes succeed individually, and two seeds do not establish broad
statistical superiority. The seed-0 development and CI cases can be rerun with `--seeds 0`.
