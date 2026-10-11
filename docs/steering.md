# A brain that reads itself

This guide implements sequential self-readback: a belief patch reads the cortex's
diagnostics and returns sensory gains for the cortex's next finite repair.
The **steered cortex** has senses weighed by a steering patch. The **governor**
reads the brain's own signals and decides when to think, and the **life** runs
both. These compose existing belief models and a settling graph governor with
an explicit learning interface and admitted parameter steps.

**This composition does not jointly settle observer and observed activity.**
The observer sees a recorded readback, computes its output, and the cortex
then repairs under that output. Joint parameter admission checks a proposed
weight step, not a common fixed point. A recursive network with one equilibrium
must instead include all observing states and their feedback in the same
equations and stopping check. The [contract guide](contracts.md) makes this
distinction explicit.

Under the flagship theorem's linear/additive assumptions, a tower of readbacks
can be absorbed into one effective map. Nonlinear observation, gain control,
restricted sensing and a computation budget create distinctions worth testing;
they do not by themselves establish useful depth. The claim needs a task
advantage at matched information and resources: compare the step on, the step
off and a hand-designed control, then test against strong matched conventional
models.

## The steered cortex

`Steered(cortex, steering, weighing)` puts a second `BeliefPatch` over the first. Each moment,
the cortex's readback is assembled into a vector the steering patch observes, the steering
patch runs its finite repair, the weighing turns its outputs into a gain per block of the cortex's port, and
the cortex assimilates the moment under those gains. The readback's channels, in order:

| channels | what they carry |
| --- | --- |
| `probe:b` | the repair map's move from the expectation with only block `b` heard (the residual-alone probe) |
| `surprise:b` | block `b`'s reading against the reading the previous belief implies, in persistence units |
| `residual` | the previous moment's repair residual |
| `output:i` | the previous moment's outputs, with `reads_output=True` |
| `evidence:b:i` | the encoded evidence of block `b` before any gain, for the blocks in `evidence` |
| `extra:i` | channels a callable supplies, with `extra=` and `extra_channels=` |

Which channels the steering patch hears is the mask of its port, a gene: the brain below
hears the surprises and not the evidence, and the mask says so without touching the weights.

```python
import numpy as np
from cadence import BeliefPatch, DenseBlock, Softmax, Steered, StructuredPort

# a dummy at a position in [0, 1]^2 reported by an eye and an ear; in one phase the ear is displaced
rng = np.random.default_rng(0)
def world(n, t, displaced):
    x = np.cumsum(rng.normal(0.0, 0.05, (n, t + 1, 2)), axis=1) % 1.0
    eye = x[:, :-1] + rng.normal(0.0, 0.01, (n, t, 2))
    ear = x[:, :-1] + (0.3 if displaced else 0.0) + rng.normal(0.0, 0.01, (n, t, 2))
    return np.concatenate([eye, ear], axis=-1), np.zeros((n, t, 1)), x[:, 1:]

eye, ear = DenseBlock(0, 2, 6), DenseBlock(2, 2, 6)
cortex = BeliefPatch(StructuredPort(4, [eye, ear]), actions=1, belief=12, outputs=2, cells=64, active=4, record_width=8, seed=1)
def implied(y):                       # what each sense should read if the belief is right
    r = np.full((len(y), 4), np.nan)
    r[:, :2] = y                      # the eye reads the position
    return r
cortex.set_implied_reading(implied, units=[0.0025, 0.0025])   # the persistence error of the compared channels

# the steering patch hears the two surprises and the residual, not the probes: 2 + 2 + 1 channels
hears = np.array([False, False, True, True, True])
steering = BeliefPatch(StructuredPort(5, [DenseBlock(0, 5, 4)], mask=hears), actions=1, belief=6, outputs=2, cells=16, active=2, record_width=4, seed=2)
brain = Steered(cortex, steering, Softmax(2, span=2.0), rate_scale=0.5)
assert brain.channels == 5 and brain.readback_names == ["probe:0", "probe:1", "surprise:0", "surprise:1", "residual"]
assert not brain.probes_on                                   # unheard probes are not computed
```

Learning is within the life. With a target and a rate, `run` takes the cortex's adjoint (its
deltas and the gradient of its loss into the gains), pulls that gradient through the weighing to
the steering patch's outputs, takes the steering patch's adjoint under it, and admits one joint
step by a replay of the chunk: the steering patch replayed on the recorded readbacks, taken as
given, the cortex under the gains it then returns, the step halved while the objective does not
fall by the Armijo margin, starting from twice the last admitted step and at most `rate`.

```python
for phase in range(6):
    o, a, y = world(4, 16, displaced=bool(phase % 2))
    taught = brain.run(o, a, y, rate=4.0, state=brain._fresh(4))
    assert taught.updated and taught.steering_updated and taught.reason == "updated"
o, a, y = world(4, 16, displaced=True)
path = brain.run(o, a, y, state=brain._fresh(4))
assert np.allclose(path.gains.sum(axis=-1), 2.0)          # the weighing moves weight between the senses
assert path.gains.shape == (4, 16, 2) and path.readback.shape == (4, 16, 5)
assert brain.step_size is not None and brain.cost["replays"] >= 6
```

`SteeredPath` carries the outputs, the gains, the readbacks, the cortex's residuals, the
steering patch's outputs, the loss and the price, what the step did (`updated`,
`steering_updated`, `step`, `halvings`, `replays`, `reason`), and the last moment's two
`BeliefPath`s for a page. `run(learn_cortex=False)` lets the cortex sleep: its parameters stay
and only the steering patch steps, its gradient still the cortex's. Learning an exception
needs that regime: with the cortex learning alongside, its own adaptation absorbs the
exception the steering patch is trying to learn.

## The weighing

The weighing is the application's declaration of what the steering patch may do to the cortex.
`Softmax(blocks, span)` returns gains that sum to the number of blocks, `blocks * softmax(span *
tanh(y))`: weight moves from one sense to another and no sense is amplified against the belief.
`Gaze(blocks, sigma=, cut=, lamp=, span=, price=)` is a window over a ring of blocks whose
centre the steering patch turns by `span * tanh(y)`: each block's gain is a Gaussian in its
distance from the centre, cut beyond `cut` widths, at a brightness that falls as the window
widens, and the turn carries a price in the objective the joint step is admitted on. The centre
is the weighing's state and travels in the boundary. `Rule(fn)` is a hand-written map from the
readback to the gains, the control: no steering patch, no learning of the rule, the
cortex under it learning by its own admitted step. A weighing declares `gains(y, state)` and
`pull(ys, gains, dgains, states)`, the Jacobian's action on the gradient into the gains; a new
one is those two methods.

```python
from cadence import Gaze, Rule

ring = BeliefPatch(StructuredPort(8, [DenseBlock(i, 1, 1) for i in range(8)]), actions=1, belief=8, outputs=8, cells=16, active=2, record_width=4, seed=3)
lookout = BeliefPatch(StructuredPort(17, [DenseBlock(0, 17, 4)]), actions=1, belief=4, outputs=1, cells=16, active=2, record_width=4, seed=4)
beam = Steered(ring, lookout, Gaze(8, sigma=0.5, cut=2.0, lamp=0.3, span=0.4, price=0.01))
sea = rng.random((2, 6, 8))
lit = beam.run(sea, np.zeros((2, 6, 1)))
assert lit.gains.shape == (2, 6, 8) and (lit.gains == 0.0).any()          # a window, cut beyond two widths
assert beam.boundary().weighing.shape == (2,)                              # the centre travels in the boundary

def rule(readback):                                                        # the hand-designed control
    ear_loud = readback[:, 3] > 2.0
    ear = np.where(ear_loud, 1.8, 0.4)
    return np.stack([2.0 - ear, ear], axis=-1)
control = Steered(cortex, weighing=Rule(rule, macs=6))
ruled = control.run(o, a, y, rate=4.0, state=control._fresh(4))
assert ruled.steering_output is None and control.macs_per_moment() == cortex.macs_per_moment() + cortex.readback_macs(probe=False) + 6
below = Steered(cortex)                                                    # fixed gains of one: the brain below
assert np.all(below.run(o, a, state=below._fresh(4)).gains == 1.0)
```

## The switch and the accounting

The three arms are three constructions of the same class: `Steered(cortex, steering,
weighing)` with the step on, `Steered(cortex)` or a steering patch deaf to the tested channel
with the step off, and `Steered(cortex, weighing=Rule(...))` as the control. Two ablations act
on a trained brain at test time: `brain.ablation = "cut"` freezes the gains at one while the
steering patch still runs and is counted, a callable ablation maps the gains (a shuffle,
`lambda g: g[:, ::-1]`); `brain.deaf = mask` zeroes readback channels. Every
arm reads one accounting: `macs_per_moment()` (the cortex's moment with the probes when read,
the steering patch's moment, a rule's declared operations), `moments_per_decision()`, and
`cost`, the two patches' counters summed with the joint admission's replays and
actually executed current-moment readbacks. These are dense forward-work estimates;
adjoints, selection, callbacks and non-MAC operations require separate accounting.
A zero gain does not make the current dense port kernel sparse.

```python
brain.ablation = "cut"
cut = brain.run(o, a, state=brain._fresh(4))
assert np.all(cut.gains == 1.0) and cut.steering_output is not None
brain.ablation = None
brain.deaf = np.array([True, True, True, False, True])   # deaf to the ear's surprise
assert np.all(brain.run(o, a, state=brain._fresh(4)).readback[:, :, 3] == 0.0)
brain.deaf = None
assert brain.macs_per_moment() == cortex.macs_per_moment() + cortex.readback_macs(probe=brain.probes_on) + steering.macs_per_moment()
```

## A life lived online

A page steps a brain one moment at a time and learns from a chunk it has already shown.
`boundary()` returns a detached copy of where the brain is between two moments (both beliefs, the previous output and
residual, the weighing's state, the moments seen); `run(state=boundary)` starts there, and
`keep_live=True` leaves the live boundary where it was. Keep the boundary before a chunk, step
through it, then learn from it: the replay makes the same moments the steps made.

```python
brain.reset()
kept = brain._fresh(1)
brain._live = kept
stream = o[:1]
seen = np.concatenate([brain.run(stream[:, k : k + 1], a[:1, k : k + 1]).gains for k in range(16)], axis=1)
learned = brain.run(stream, a[:1], y[:1], rate=0.5, state=kept, keep_live=True)   # the same moments, one step
assert np.allclose(learned.gains, seen) and brain.boundary().moments == 16
```

## A life with a governor

`Life(patch, governor, ...)` is one continuing loop in which a governor of the same rule
reads the brain's own signals and returns its mode. The life computes a readback of seven
channels from the patch's signals, the last surprise over its baseline (log-compressed), a slow
average of the same, the repair residual over its routine median, the last mode as three flags
and a constant; the governor settles on it and names the mode. **Habit** is the application's
cheap policy. **Imagine** evaluates the application's candidate actions over a horizon in the
belief's private continuation and executes the best first action. **Learn** replays the executed
window from its boundary and takes one admitted step per pass, undoing a window whose loss did
not fall by the validity margin. The baseline of the surprise follows the quiet moments with a
floor, and habituates when a learn call was undone. The governors: `PatchGovernor`, a settling
brain of readback, cortex and motor neurons whose every synapse is a gene, with a hand-set
wiring as the founder genome and `PatchGovernor.space()` for `genes`; `ThresholdGovernor`, the
hand-designed control; `AlwaysAwake` and `NeverWakes`, the two ends of the switch.

```python
from cadence import Life, LifeConfig, PatchGovernor, ThresholdGovernor

class Sill:                      # a dot that drifts and a paw the action moves; the drift flips once
    def __init__(self, seed=0, change=60):
        self.rng, self.x, self.p, self.t, self.change = np.random.default_rng(seed), 0.5, 0.5, 0, change
    def reading(self):
        return np.array([self.x, self.p])
    def step(self, a):
        self.t += 1
        drift = 0.01 if self.t < self.change else -0.03
        self.x = float(np.clip(self.x + drift + self.rng.normal(0.0, 0.001), 0.0, 1.0))
        self.p = float(np.clip(self.p + 0.1 * float(a[0]), 0.0, 1.0))
        return self.reading()

def make_life(governor, seed=0):
    patch = BeliefPatch(StructuredPort(2, [DenseBlock(0, 2, 6)]), actions=1, belief=8, outputs=2, cells=32, active=4, record_width=4, seed=seed)
    return Life(patch, governor,
                habit=lambda r: np.array([0.5 * (0.5 - r[1])]),                  # the paw drifts home
                propose=lambda r: np.array([[-1.0], [0.0], [1.0]]),              # three pushes to imagine
                advance=lambda r, y: r + y,                                       # a predicted change moves the reading
                cost=lambda R, A: (R[:, 0] - R[:, 1]) ** 2 + 0.01 * A[:, 0] ** 2, # the task's price
                target=lambda r, r2: r2 - r,                                      # what the brain should have predicted
                baseline=1e-4, residual0=0.1,
                config=LifeConfig(window=16, min_window=8, min_cooldown=4, passes=3, learn_rate=2.0, horizon=3))

for governor in (PatchGovernor(), ThresholdGovernor({"k_imagine": 3.0, "k_learn": 1.5, "persist": 6, "persist_share": 0.5, "cooldown": 0})):
    world, life = Sill(), make_life(governor)
    reading = world.reading()
    for _ in range(90):
        reading, decision = life.step(reading, world.step)   # decide, act in the world, close the moment
    report = life.compute()
    assert sum(life.totals["decisions"].values()) == 90 and report["moments_per_decision"] >= 1.0
    assert report["learn_calls"] == report["kept"] + report["undone"]
```

`life.records` holds every decision (mode, action, expectation, residual, readback, governor
steps, imagined moments, surprise, baseline, target), `life.learns` every learn call (window,
loss before and after, valid, kept, passes), and `compute()` the cost: the patch's moments and
multiply-accumulates, the governor's steps as moments, per decision. A `Steered` cortex is a
patch too: under a governor its steering patch learns with the joint step, and a governor
gating the cortex's own learning is this composition.

## The instruments of the orienting response

A gain's response to an event is measured with `orienting(gain, events)`:
per event the capture (the gain's peak during the event over its level in the quiet frames
before), the latency (the first frame from the onset at which half the capture is reached) and
the return (frames after the event until the gain is back within a fifth of the capture); per
kind the capture of the first and the last events, the habituation curve and the latency shares.
`dishabituation(rows, consequential=, kind=)` compares one kind's captures just before and just
after a consequential event.

```python
from cadence import dishabituation, orienting

gain = np.full(120, 0.5)
events = []
for i in range(6):                              # a clock whose capture falls
    gain[10 + 15 * i] = 0.5 + 0.4 * (1 - i / 6)
    events.append(("clock", 10 + 15 * i, 1))
gain[100:103] = [0.6, 0.9, 1.0]                 # a cry
events.append(("cry", 100, 3))
measured = orienting(gain, events, pre=4, post=12, bins=3)
clock, cry = measured["kinds"]["clock"], measured["kinds"]["cry"]
assert clock["capture_first"] > clock["capture_last"] and cry["capture_mean"] > 0.4
assert measured["rows"][-1]["latency"] == 1
```

## What it does not do

The replay the joint step is admitted on takes the recorded readbacks as given: the steering
patch is not re-run under the proposed cortex, as in the seam. The gaze's pull is local in time;
the dependence of later centres on an earlier turn is not propagated. The surprise of a
stream's first moment compares against the readout of the boundary belief. The adjoint's cost is
not in the counters (a backward scan costs about two forwards). The cortex's store is left unwritten by default (`observe(write=False)`). The life's
imagination is the application's proposal and cost; the library carries the loop, the governor
and the accounting, not the task.

## Where to go next

[How to build a demo](howto-demo.md) says how a demo accepts one of these mechanisms: the
three-way switch, the day and night regimes, the measurements, the selection and the
receipts.
