# Building on Cadence

Every change must meet [the manifesto](MANIFESTO.md) and its
[contribution checks](CONTRIBUTING.md). Keep those documents in sync when a
requirement changes, and distinguish design goals from implemented capabilities.

Start with [the guided documentation README](docs/README.md), then read the
[continuing-world-model guide](docs/world-model.md) before designing an application,
tutorial or experiment. Use [the catalogue](docs/index.md) to find individual
guides. Read the owning guide, [numerical contracts](docs/contracts.md) and
[API reference](docs/api.md) before changing semantics.

| Task | Read and use |
| --- | --- |
| Build a continuing brain | [Quickstart](docs/quickstart.md), [composition](docs/brain.md), [architecture](docs/architecture.md), [runnable lifecycle](examples/continuing_brain.py) |
| Manage streams, outcomes and memory | [Continuous interaction](docs/continuous.md), [memory](docs/memory.md), [reward](docs/reward.md), [experience](docs/experience.md) |
| Diagnose learning or settling work | [Learning](docs/learning.md), [contracts](docs/contracts.md), [API diagnostics](docs/api.md), [troubleshooting](docs/troubleshooting.md) |
| Use retained-record dreaming and sleep | [Record patches](docs/record-patch.md): `RecordPatchNet.dream` and `sleep`; this is a separate model from `Brain.compose` |
| Learn transitions, plan or protect responses | [Temporal patches](docs/temporal.md), [planning](docs/planning.md), [temporal memory](docs/temporal-memory.md), [runnable example](examples/memory_imagination.py) |
| Add optional recursive observation | [Cortical regions](docs/cortex.md), [recursive settlement](docs/recursive-settlement.md), [recursive training](docs/recursive-training.md) |
| Work on the population equilibrium solver | [Its entry guide](docs/equilibrium/README.md), [agent instructions](docs/equilibrium/AGENTS.md) and [reference](docs/equilibrium/REFERENCE.md) |
| Change or release the library | [Contributing and required checks](CONTRIBUTING.md), [task design](docs/task-design.md), [protocols](docs/protocols.md), [receipts](docs/receipts.md) |

## What Cadence is, and is not

We are building an animal-like brain. An independent input/label classifier,
an MLP or a transformer does not establish the intended Cadence lifecycle.
A feed-forward patch chain with an external trained answer head is a control;
the default reciprocal composition already has a different settlement and
memory contract, even with one processing region.

The intended brain bootstraps a useful equilibrium world model, carries state
and memory through a continuing life, and repairs witnessed failures locally.
Low routine work remains a measured target; a small residual does not establish
low physical energy or efficient task performance. Build and judge the brain
through streams, retained state, witnessed corrections, free behavior over time,
retention and recovery, with backpropagation networks as matched baselines.
Keep supported actual-reward updates distinct from an application's selective
teacher policy. `step` learns from every outcome. `Brain.live` is the explicit
routine-and-repair loop of one stream: a brain constructed with arousal genes
answers greedily and learns nothing while calm, and samples, learns and writes
memory when an outcome surprises it or reward stays below its usual level. Its
evidence is one bounded chamber; a general failure-only learning gate is not
implemented.

## Goal and mechanism

**Fixed foundation and default hypotheses.** Every addition must remain within
local agreement repair between patches into the same coupled global equilibrium.
Try the simplest existing System 1 first. Check state, experience, available
information, memory and implementation before adding a mechanism. Animal and
human brains are the evolved functional reference; build abstracted solutions,
with biological studies suggesting tests rather than supplying proof. Finite
capacity, rigid learned interpretations and unfamiliar-environment failure are
possible limits, not a promise of perfect learning. Follow the mandatory
[mechanism and capability-preservation review](CONTRIBUTING.md#preserve-the-capable-foundation)
before promoting a change. Preserve demonstrated recall, associations, context,
actual-action ownership, private imagination and saved continuation.

Cadence aims to build a simulated human-like brain from simplified biological
mechanisms. **System 1 is the default:** a continuing animal-like brain with
memory, plasticity, private imagination and action. **System 2 is optional:**
observing cortical regions add recursive feedback in that same neural graph.
Base modules can already be deep and specialized; ordinary depth is not recursive
observation.

The organizing lifecycle is **bootstrap a useful reciprocal interpretation and
memory → use it → witness a failure → repair locally → continue the same brain**.
The architectural hypothesis is that the learned equilibrium is the world model
in operation. Learned parameters and memory support a family of equilibria under
changing evidence, not one permanently fixed activation. Preserve one acquired
brain across normal use, interference and correction. Biological names are
functional software roles, not claims of literal biology.

Lead applications and tutorials with that continuing lifecycle. A fresh brain
per observation, memory-bypassing classification, and an external trained answer
readout are controls or separate models, not the flagship brain demonstration.
Keep isolated learning and calibration controls, clearly labeled, because they
test mechanisms the composition still needs. Do not remove working capabilities
or blur model identities to make the design story simpler.

The intended observer-like structure has bounded local state, sensory/action
boundaries, inspection, retained evidence and returning constraints. Name the
actual implementation: composed brains expose sensory/motor indices, neural
state, a working trace and associative memory. These are not the typed `Ports`
or record readback of the record/temporal APIs, nor the population solver's exact
state-and-error readback. A retained trace may be a held boundary during a present
solve; a live coordinate must still qualify. Do not freeze unresolved live state
or weaken qualification to manufacture a whole-brain answer.

## Build and demonstrate the lifecycle

1. Declare observations, action meanings, stream identities and the environment's
   outcome signal. State which state persists and which boundaries are held.
2. Compose one reciprocal System 1 brain and bootstrap useful behavior. The
   default already has recurrence and memory; a layered layout does not make it
   feed-forward. Add depth or optional observers for a task requirement, not as a
   substitute for testing the lifecycle.
3. Keep that brain alive during interaction. For one creature, construct it with
   `arousal=True` and use `live` to read memory, execute its action and
   report the actual outcome exactly once. Use `step` for explicit teaching,
   batched streams or learning from every outcome; targets label the current
   observation, while rewards describe the preceding executed action.
4. Introduce a disturbance, detect a witnessed mismatch or failed objective,
   and apply supported learning. Keep numerical settlement separate from durable
   repair. Do not silently redefine `learn=True`, reward feedback or memory
   writes as an automatic failure-only gate; `live` is the supported loop in
   which the brain's own arousal makes that decision.
5. Measure recovery and earlier skills with answers free, alongside normal and
   repair work. Save and reload the same brain, including pending feedback, and
   verify its continuation. Charge rehearsal, imagination and refused work.

## Keep the main interface simple

Lead user guides with `Brain.compose(..., arousal=True)` and `brain.live(...)`
for one continuing brain. Keep `step` and separate `act`/`learn` as explicit controls,
and `last_settlement` as diagnostics. Use `NeuralGraph` for the lower-level
neural graph. `modules` selects region sizes; adjacent regions and the
association/motor pair exchange signals, while sensory inputs supply a held
drive. Optional `observers` read and return to that same graph. Working trace
and consolidating associative memory are included. Add a new abstraction only
for a demonstrated general need; changing an interface must preserve the
behavior it serves. Set individual genes with `learning_*`, `actor_*`, `arousal_*`
and the named memory options in `compose`; use `brain.retune(...)` to change them
in the same acquired life. `brain.describe()` reports the effective settings.
Keep rate ownership explicit: actor rates govern reward learning, learner rates
govern teaching, and temperature is shared. A fresh configuration dataclass is
still a complete replacement, while named overrides preserve unspecified values.
Consult the [defaults and tuning guide](docs/brain.md#defaults-and-expert-overrides).
Application recipes are measured choices, not generally optimal defaults.

`resting_bias` is an optional initialization setting, not evidence of improved
acquisition or retention. Keep its default and the working-trace defaults as
controls. Select alternatives with matched tasks and continuation tests;
responsiveness under random drives or one tiny teaching assay cannot establish
generally better defaults. The [recall instrument](benchmarks/recall/README.md)
and [acquisition microscope](benchmarks/acquisition/README.md) have distinct
protocols. Preserve their measured settings, failed attempts and source identity.

`step` receives a current observation and the preceding executed action's actual
outcome. `teacher` labels the current observation. Preserve stream identity,
event order and pending feedback. `imagine` evaluates supplied observations
using a private trace and read-only durable memory; it does not predict the
world's transitions or turn predictions into witnessed experience.

Distinguish implemented contracts from the desired world-model lifecycle.
`Brain.compose` does not yet integrate learned environmental transition prediction.
`step` does not gate learning on witnessed failure; `live` gates it by arousal
for one stream through its [arousal law](docs/continuous.md#routine-and-repair-live).
Youth and sustained arousal also allow learning from successful outcomes. Its
routine moment still pays one full settle. Stable inputs do not guarantee cheap
operation. Do not invent thresholds, success policies or runtime
changes to make documentation imply those capabilities. Missing integration and
behavioral contracts need their own capability issues and tests.

Advanced APIs have explicit contracts: `TemporalPatchNet.plan` uses a learned
world model; `TemporalMemory` protects selected responses at finite capacity;
record patches combine context, learned relations and writable records. The
population solver under `cadence.experimental.equilibrium` provides exact
state-and-error readback. Do not transfer its mathematical guarantees to a
neural-graph or record operation without an actual correspondence.

Preserve `RecordPatchNet.dream(inputs)` and `sleep(cues)`: dreams complete cues
from retained records without learning; sleep fixes those targets, teaches slow
weights and rewrites record residuals against the changed weights. This retained
capability is separate from `Brain.imagine` and online `SynapticMemory`
consolidation; `Brain.compose` does not expose the record-patch sleep cycle.
Dreamed targets are not witnessed environmental outcomes. Use the owning guide's
model and source-pinned evidence when reproducing sleep results.

Graph and temporal learners use free/nudged equilibrium contrasts. Record and
belief models also use explicit adjoints and record writes. State those
mathematics accurately. A record scan is not a joint graph-equilibrium certificate.

## Numerical and behavioral contracts

`Brain.act`, `predict` and `accuracy` check the full state equations through
`NeuralGraph.equilibrate`, including cached activity and optional observers. Refused
`act` calls preserve live state, memory, randomness and pending feedback. If
`step` learns a real outcome before its next action refuses, that learning stays:
retry `act`, not the reward. Finite nudged eligibility and default teaching phases
retain their own contracts. Qualified supervised learning is explicit through
`LearnerConfig.qualified`: all attempted free and teaching states must meet the
original equations before an update. Refusal preserves parameters and optimizer
history. Reward eligibility has its own `ActorCriticConfig.eligibility_steps`;
the Brain default is 12, even when teaching requests longer phases.

Qualified free solves may use numerical damping within their one declared budget,
then check the undamped model's residual. This does not change the live model or
finite teaching law, and it is not System 2. A numerical qualification does not
prove correctness about the world, reliable recall or a cognitive advantage.
Keep equation residuals distinct from prediction errors and failed task outcomes.
A small residual does not measure physical power or establish transformer-level
quality, scalability or efficiency; those require matched behavioral and resource
measurements over bootstrap, normal operation and repair.

`Brain.last_settlement` reports the latest completed free-answer solve, including
refusal. `step` exposes its final `act`; scoring exposes the final `predict`
batch. Its residuals and sweep/check counts exclude feedback, teaching, reward
eligibility, imagination and memory work. Use `last_learning` and opt-in
`record_settlements` where appropriate; none is a total-work or energy meter.
Diagnostics are observational and are not checkpointed. Preserve transactional
state even when a refused attempt updates its diagnostic report.

`predict` and `accuracy` omit working and associative memory; `act` reads them.
Test graph plasticity, working traces and consolidated records separately, then
test their composition. Current teacher labels update the graph; the reward loop
writes the actually executed action's observed outcome. Clearing transient state
does not erase durable knowledge. Retention tests must use the same acquired
brain after competing experience and charge any rehearsal.

An unsuccessful reward bootstrap must preserve pending feedback and restore all
memory changes from that attempt. An accepted outcome remains learned if the
following action refuses. Keep those two retry cases distinct in tests and guides.

Memory and imagination are implemented capabilities to preserve. Test actual
acquisition, free recall, interference, private-state isolation and saved
continuation when changing them. Finite storage and supplied protection or
salience do not establish general lifelong retention.

## Review and verification

- Prefer fewer concepts, direct interfaces and working defaults. Simplification
  must preserve supported memory, learning, imagination and action behavior.
- Keep signatures, mutation rules, units and timing aligned with code. Provide
  runnable examples and useful refusal/retry behavior.
- Run [contributing checks](CONTRIBUTING.md), executable documentation and local
  links. Test installed wheel/sdist behavior as well as the source checkout.
- Preserve capability tests when changing composition or simplifying APIs:
  [acquisition](tests/test_composed_acquisition.py),
  [feedback transactions](tests/test_feedback_transaction.py),
  [memory and continuation](tests/test_generic_memory_checkpoint.py),
  [private imagination](tests/test_composed_brain.py),
  [record recall and sleep](tests/test_record_patch.py), and
  [settlement diagnostics](tests/test_settlement_reports.py).
- Numerical changes need independent reference, derivative or adversarial checks.
  Measure behavior after teaching with answers free, and count query, training,
  replay, planning and refused work. Preserve failed runs and evidence.
- A passing suite is not a blanket performance guarantee. Claims of unchanged
  or improved performance need before/after behavior and cost comparisons with
  matched tasks, inputs, seeds, information and resource accounting. Keep working
  baselines and acquisition/retention thresholds when refactoring.
- Changes to the population solver need `tests/equilibrium`; default neural
  changes need their own tests. Optional backends must preserve the applicable
  qualification and continuation rules.

Cadence is experimental: backward compatibility is not a design requirement.
Save/load must still preserve the current supported model's complete continuation.
Keep GPL-3.0 and required source attribution. NumPy is required; optional backends
must not become hidden default dependencies.

Use individual GitHub issues for missing or untested capabilities and optimization.
Optional System 2 can ship without a demonstrated task advantage. Release checks
cover correctness, capability preservation, continuation and installed artifacts;
completed general cognition is not a release gate.

## Related physics project

[Observer Patch Holography](https://github.com/FloatingPragma/observer-patch-holography)
(OPH) investigates local consistency constraints in physics. Connections and
ports provide a useful analogy to overlapping descriptions, but a mathematical
correspondence must identify the variables, equations and assumptions of each
model. In the default neural brain, qualification checks the declared state
equations. The experimental population model qualifies stationarity of its
energy and can retain nonzero prediction errors; neither forces all patches to
the same value ([prediction and disagreement](docs/equilibrium/ELEMENT.md#prediction-and-disagreement)).
Keep the two projects' claims separate. OPH results do not establish a Cadence
capability, Cadence measurements do not test OPH physics, and Cadence code and
documentation do not depend on OPH.
