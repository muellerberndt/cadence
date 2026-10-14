# Training a recursive controller

A recursive controller reads its inputs, settles its hidden and observing
populations together, and answers through output neurons in that same state.
Training then detunes the observed outputs and compares the settled phases.
This page shows the data and numerical boundaries needed to use that mechanism.
The runnable examples require only NumPy and Cadence; their tiny synthetic data
check the API, not task performance. The full controller and training loop are
marked illustrative and are not executed by documentation CI.

## What the Doom experiment established

In a verified ViZDoom **Basic** experiment, a controller with
`PatchNet.recursive(1371, [24, 8], 6)` learned useful visual actions from returns
measured by a simulator teacher. Its local-EP endpoint killed the target in all
64 reserved confirmation episodes. Blank images reduced that to 0/64 and images
from other episodes to 33/64. MLP and CNN controls also achieved 64/64; the CNN
used fewer game tics. Game tics are not inference runtime.

The teacher supplied six counterfactual action returns at each of 256 training
contexts. This was supervised visual control, with explicit recent-action
history. It establishes neither learning from sparse reward alone nor a benefit
from recursive depth, generic training efficiency, or competence on other Doom
scenarios or Atari. The original regression-fit gates remained unmet; the
separately frozen gameplay test evaluated fixed endpoints without changing those
verdicts. These are historical experiment results, not a benchmark reproduced
by this guide. The original controller checkpoints and confirmation trajectories
are not bundled with this package; the runnable examples below establish only
the current API contract. Do not use them to claim a replication of those scores.

## Choose the graph and numerical policy

The 1,371 inputs are 1,350 grayscale pixels (45 wide by 30 high) and 21 history
fields: the previous three executed actions, each encoded as six one-hot values
plus a validity flag. History is supplied input, not evidence of learned temporal
memory. `[24, 8]` means 24 base hidden neurons and one population of eight
observers. Six **distinct output neurons** represent wait, strafe left, strafe
right, fire, left-and-fire, and right-and-fire.

Every observer contacts the evolving input, base and output neurons in both
directions. All 1,409 potentials settle together; there is no learned answer
head outside the graph. Input values supply fixed soft drives, while the input
neurons themselves receive recurrent feedback. This graph has 88,512 directed
contacts, or 44,256 reciprocal weight pairs, and 1,409 biases: 45,665 independent
trainable scalars. The layer names confer no priority over other populations.

```python
import numpy as np
import cadence as cd

learning = cd.LearnerConfig(
    nudge="quadratic", centered=True, beta=0.02,
    eta=0.001, eta_bias=0.001,
    momentum=0.9, normalize=0.999, normalize_floor=1e-8,
    scale_cap=8.0,
)
batch_rng = np.random.default_rng(191)
```

This recipe uses the new optional solver policy; it is not a byte-identical
replay of the frozen research implementation.

<!-- not-run: full research-sized construction; use the small executable fixture below -->
```python
net = cd.PatchNet.recursive(
    1371, [24, 8], 6, seed=83, coupling=1.8, config=learning,
    backend="cpu", steps=2048, chunk=8, tolerance=1e-8,
    solver="hybrid", refinement_steps=64, source_capacity=0,
)
assert net.brain.connectome.n == 1409
assert net.brain.connectome.synapses == 88512
assert net.learner.parameters() == 45665
```

`hybrid` first uses the ordinary local iteration, then attempts bounded warm
Newton refinement only for unresolved rows. Supported models use CPU float64,
smooth `tanh(v/2)` neurons, reciprocal effective weights, no adaptation, and an
independent input population (zero effective input-to-input weights). The factory above
satisfies these structural conditions; nudges and temporal anchors must also
leave the eliminated input ports unanchored. Every admitted row must meet the full
equation tolerance and positive local energy-curvature check. Dense curvature
and refinement operations are global numerical work; the parameter credit from
EP remains the local phase contrast. There is no cold saddle escape or guarantee
of a global minimum, uniqueness, or a shared smooth branch between phases.

The default remains `solver="local"`, whose admission checks only the full
residual. Use it when that is the intended contract. The hybrid policy is a
research-informed option with additional costs and narrower model support.

For `I` input neurons and `C` other neurons, refinement stores blocks of size
`I × C` and `C × C`; each row's curvature construction costs roughly
`O(I C²)`, followed by `O(C³)` dense algebra when needed. Here `C = 38`,
including the six output neurons. Larger observer populations increase that
cost sharply. This is not a scalable default for every graph: compare measured
memory and latency as well as local sweep counts before increasing widths.

At initialization, `coupling=1.8` sets the maximum absolute incoming weight sum
to 1.8. The factory's activation slope is bounded by 0.5 and its default local
step is 0.5, giving a free-step contraction bound of 0.95. Learning can destroy
that certificate; an individual weight cap of 8 does not preserve a row-mass
bound. Inspect `cd.certificate(net.brain)` on current parameters, and still
qualify every free and nudged phase. See [settling certificates](certificate.md).

## Normalize observations and returns

Fit pixel statistics on training observations only, then freeze them for
validation and action. The experiment used pixels in `[0, 1]`, a per-pixel
standard-deviation floor of 0.05, and drive amplitude 2. History flags remained
unstandardized. Do not fit statistics on validation episodes or silently erase
the validity flags for missing history.

```python
# Synthetic observations only: replace these with witnessed training frames.
fixture_rng = np.random.default_rng(7)
training_pixels = fixture_rng.uniform(size=(4, 30, 45))
training_history = np.zeros((4, 3, 7))  # no preceding actions in this fixture
pixel_mean = training_pixels.mean(axis=0)
pixel_scale = np.maximum(training_pixels.std(axis=0), 0.05)


def encode(pixels, history):
    pixels = np.asarray(pixels, dtype=np.float64)
    history = np.asarray(history, dtype=np.float64)
    if pixels.ndim != 3 or pixels.shape[1:] != (30, 45):
        raise ValueError("expected (batch, 30, 45) pixels")
    if history.shape != (len(pixels), 3, 7):
        raise ValueError("expected (batch, 3, 7) action history")
    if not np.isfinite(pixels).all() or not np.isfinite(history).all():
        raise ValueError("observations must be finite")
    if np.any((pixels < 0) | (pixels > 1)):
        raise ValueError("pixels must already be scaled into [0, 1]")
    normalized = ((pixels - pixel_mean) / pixel_scale).reshape(len(pixels), -1)
    return np.concatenate((normalized, history.reshape(len(pixels), -1)), axis=1)


features = encode(training_pixels, training_history)
assert features.shape == (4, 1371)
```

The teaching values were measured 36-tic action returns divided by 300; the
policy later chose an action for at most 12 tics before observing again. These
are task choices, not universal return scales. Fix the scale before evaluation,
keep targets within the useful range of the bounded output activity, and report
any clipping as a change to the target. Never insert a zero return for an action
whose outcome was not observed.

`observe` accepts a boolean `observed` mask shared across the batch's output
ports. A full counterfactual teacher can mark all six true. If each record
contains only one executed action's return, form minibatches with the same
observed action (or process one record at a time). Grouping changes the update
sequence and must be part of the declared training protocol. The current API
does not accept a different output mask per row. Unknown entries may be NaN;
putting free predictions there and marking them observed would add constraints
during the nudged phases.

For batch size `B`, output activities `s`, targets `y`, shared mask `m` and
optional nonnegative row weights `w`, the quadratic objective is

`L = (1 / (2 B)) * sum_b w[b] * sum_o m[o] * (s[b, o] - y[b, o])**2`.

It is a **half-sum over observed outputs, averaged over batch rows**. It is not
divided by the observed-port count or by the sum of weights. Match this objective
when checking an implicit gradient or training a comparison model. With all six
ports observed and unit weights, `L = 3 * mean_squared_error`; report which
quantity a curve shows. Row weights are externally supplied teaching gains.

```python
def quadratic_loss(prediction, target, observed, weight=None):
    error = prediction[:, observed] - target[:, observed]
    gain = np.ones(len(prediction)) if weight is None else np.asarray(weight)
    return float(0.5 * np.mean(gain * np.sum(error**2, axis=1)))


observed = np.array([False, False, False, True, False, False])
target = np.full((2, 6), np.nan)
target[:, 3] = np.array([30.0, -15.0]) / 300.0  # illustrative witnessed returns
assert np.isclose(quadratic_loss(np.zeros((2, 6)), target, observed), 0.003125)
```

## Admit phases before learning or acting

Unrelated replay examples must not inherit another example's activity. Call
`reset()` before every independently sampled minibatch and each independent
evaluation batch. This clears fast state and preserves learned parameters and
optimizer moments. Persistent row state is appropriate only when each row
actually continues the same stream; explicit action history alone does not
justify carrying unrelated row states.

```python
# A small API fixture: same rule and six output ports, no Doom training claim.
small = cd.PatchNet.recursive(
    3, [4, 2], 6, seed=83, coupling=1.0, config=learning,
    steps=256, chunk=8, tolerance=1e-9,
    solver="hybrid", refinement_steps=64, source_capacity=0,
)
small_inputs = np.array([[0.2, -0.1, 0.3], [-0.2, 0.1, 0.4]])
drive = small.stimulus(small_inputs, amplitude=2.0)
small.reset()
lesson = small.observe(drive, target, observed=observed)
assert lesson.updated, lesson.reason
assert all(np.all(p.qualified) for p in (lesson.free, lesson.plus, lesson.minus))
loss_before_update = quadratic_loss(small.read(lesson.free), target, observed)
assert np.isfinite(loss_before_update)

# Re-solve after the parameter update before choosing an action.
small.reset()
phase = small.settle(drive)
if not np.all(phase.qualified):
    raise RuntimeError(f"no action admitted: residuals={phase.residual}")
actions = small.read(phase).argmax(axis=1)
assert actions.shape == (2,)
assert phase.refinement is not None
assert all(status in ("local", "refined") for status in phase.refinement.status)
```

The target is absent from the free phase. Centered EP starts both detuned phases
from that same free state and updates only after all required rows qualify.
Check `lesson.updated` and retain `lesson.reason` and all available phases on
failure. A rejected phase leaves parameters, optimizer moments and evidence IDs
unchanged; finite free activity may remain. Do not issue a fallback action and
count it as an admitted network answer. Unsupported solver configurations can
raise exceptions, which the application must also retain as failed attempts.

`phase.converged` means residual at or below tolerance. `phase.qualified` adds
the hybrid policy's curvature and status checks. `phase.steps` counts local
sweeps. `phase.refinement` records the pre-refinement `local_residual`, accepted
Newton `steps` per row, `min_curvature`, and per-row `status`. These are useful
diagnostics, not complete work counters: they omit line-search trials, dense
matrix work and audit costs. A research comparison must count those separately.

The phases in `lesson` refer to the **pre-update** parameters. They do not
certify equilibrium after learning. Positive endpoint curvature also does not
prove EP's finite-nudge gradient accurate; check representative directions
against finite differences or an independently qualified implicit gradient.
See [learning and gradient conventions](learning.md).

<!-- not-run: requires the real training dataset and a declared experiment budget -->
```python
batch_rng = np.random.default_rng(191)
for update in range(100):  # a declared budget, not a demonstrated fit threshold
    ids = batch_rng.integers(0, len(x_train), size=32)
    net.reset()
    lesson = net.observe(net.stimulus(x_train[ids], amplitude=2.0), y_train[ids])
    if not lesson.updated:
        raise RuntimeError(f"update {update} rejected: {lesson.reason}")
```

This all-port loop requires six observed labels per row. `source_capacity=0`
allows deliberate offline replay; repeated training is not new independent
evidence. For an event stream, use the [source-ID contract](patchnet.md) instead.
Keep normalization, masks, target provenance, batch IDs, failures and budget caps
with the training record. Evaluate validation and held-out episodes without
calling `observe`, and do not retune on the held-out result.

## Save the whole application state

```python
from pathlib import Path
import json

path = small.save("recursive-controller.npz")
resumed = cd.PatchNet.load(path)
assert resumed.solver == "hybrid" and resumed.refinement_steps == 64
np.testing.assert_array_equal(resumed.brain.weights, small.brain.weights)
np.testing.assert_array_equal(resumed.state.v, small.state.v)

# Input statistics and sampling state belong to the application, not PatchNet.
np.savez("controller-inputs.npz", mean=pixel_mean, scale=pixel_scale,
         return_scale=300.0, input_amplitude=2.0)
Path("controller-rng.json").write_text(json.dumps(batch_rng.bit_generator.state))
```

`PatchNet.save` preserves parameters, optimizer, current activity, evidence-ID
window, and solver policy. Exact continuation also requires the same numerical
backend/precision, application RNG state, data order and preprocessing. Restore
the actual training RNG rather than reseeding it at resume time. Older checkpoint
formats load with the original local policy. For independent evaluation, call
`resumed.reset()` after loading. Saving does not establish useful memory.

## Scale only after controlled tests

First verify fitting on a small witnessed dataset, full-state admission, and a
useful learning curve. Compare the same inputs, history, labels, loss, validation
rules and tuning budget against a suitable MLP, CNN or transformer. Separate
parameter counts, training work, inference latency and memory; count unsuccessful
phases and teacher collection too. The Doom comparison did not establish a
compute advantage.

For a claim about recursive depth, include a wider shallow graph, account for
both reciprocal parameters and contacts, and test observer-feedback interventions
on the same learned task. A reciprocal cut changes both directions; a one-way
cut changes the energy contract and cannot silently reuse its EP guarantee.
Frozen or sequential observers test different mechanisms. More populations do
not automatically add a useful hierarchy. The
[recursive-settlement guide](recursive-settlement.md) explains the topology and
causal controls. An Atari application needs its own data, action/history design,
baselines and held-out performance evidence.
