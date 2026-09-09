# Deploying in a body

A patch net in a body is a loop: sense, clamp, settle, read, act; then, when a target or a
reward arrives, nudge and update. Nothing about that loop is a training phase. This page is
the recipe; the cart-pole and sign-writer rungs of cadence-examples are the worked examples
(a body from reward, and a body with an arm and a moving sensor).

## The loop

```python
import cadence as cd

learner = cd.Learner.load("brain.npz", backend="cpu")     # trained elsewhere; runs here
state = None
while alive:
    levels = sense()                                       # any levels in [0, 1], one per input owner
    drive = learner.engine.clamp_levels(levels[None, :])
    state = learner.free(drive, warm=state)                # a settlement, warm from the last one
    act(state.activation[0, learner.output_index])         # read the output owners, move the body
    if target is not None:                                 # a label, a level pattern, an action's advantage
        learned, report = learner.step(drive, target, warm=state)
```

- **Sense** with a place code where the quantity is a magnitude (a joint angle, a velocity,
  a pressure), with pixels where it is an image, with a filterbank envelope where it is a
  sound; levels in [0, 1], a few active owners per row. Cart-pole turns its four floats into
  a soft one-hot over bins per variable (`place_code` in its `train.py`); the sign writer
  sees pixels only, the sign and what it has drawn so far through a 7×7 window around its
  pen. `docs/tasks.md` has the recipes.
- **Settle** from the previous state (`warm=`): a body's world changes a little per frame, and a
  warm settlement takes a few steps where a cold one takes tens.
- **Read** the output owners. An action is the most active owner of its group (cart-pole:
  left or right; the sign writer: one of nine pen moves) or a draw from the softmax over
  them while exploring. A command for a muscle is best held as a population code (a bump
  over a few owners) and read out as the bump-weighted mean; the rule learns a pattern far
  better than a single owner's level.
- **Learn** when there is something to learn from: a label (`step` with class ids, as the
  sign writer learns the teacher's pen moves), a level pattern (`nudged` with a target array
  and a mask), an action's advantage (`step` with `weight=`, as cart-pole learns from its
  returns). A stream has no epochs: keep the rate constant and small, and give the seams a
  decay (`LearnerConfig.decay`) so what is not relearned fades and the net stays plastic.

## Several learners in one net

A body has more than one thing to learn: what comes next (a memory), which command makes which
sensation (a mirror), which command paid off (a critic). One net can hold them as separate
output groups on separate hidden populations. Two masks keep them apart:
`Learner(trainable_overlaps=..., trainable_owners=...)` restrict an update, and its decay, to
one population's seams and owners; set them before each learner's update, swap in each
learner's own `config`, and hand each its own `velocity` and `velocity_bias` arrays when
momentum is on, so that neither learner damps the other's running contrast.

## Checkpoints

`learner.save("brain.npz")` writes the wiring, every seam's scale, every owner's gain and bias,
the rule, the configuration, the masks, tie groups and momentum state in one file.
`cd.Learner.load(path, backend="torch", device="cuda")` rebuilds it anywhere; pass
`config=cd.LearnerConfig(eta=0.0, eta_bias=0.0)` for a deployment that settles and never
changes, or a smaller rate for one that keeps learning slowly.

## What to measure before you trust it

- **Conformance.** `cd.conformance(engine, drive, steps=60)` settles the accelerated engine
  and the owner-by-owner reference side by side and reports the largest deviation; the
  examples' receipts carry one for every net.
- **A receipt.** Bind the numbers you deploy on to the code and data that produced them
  (`docs/receipts.md`); a body that learns will drift from them, and the receipt is the
  reference you drift from.
- **Timing.** A settlement is tens of steps of one sparse or dense matrix product; on a CPU
  core a net of a thousand owners settles in a few milliseconds in NumPy and in tens of
  milliseconds in a browser, and the torch backend batches thousands of rows on an
  accelerator. Measure yours.

## What the rungs leave out, and what a body would do instead

A frame clock instead of continuous time (a body can settle warm on every sensor update); a
window of recent frames instead of recurrent dynamics (a body can carry the settled state
forward); a scripted environment instead of a physical one (the same loop, with the receipt
taken on the real sensors). Each is named in the rung's tutorial so that a later body can
replace it.
