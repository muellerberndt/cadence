# Build and test a deep brain

Teach the action, not the hidden representation. In a reciprocal brain, motor corrections
can change the hidden regions that helped produce the action. Cadence v0.104.0 fixes two
ways that correction could be lost in settling or recall learning. The fixes do not make
every architecture learn a useful division of work. That also depends on the wiring, the
observations available at the decision, and whether doing the task requires the hidden path.

## Give the hidden region a job

Start with a shallow brain and a task where it has something real to learn. Add a hidden
region to solve a specific problem, such as using an earlier cue to coordinate two outputs
whose recent inputs look alike. A useful modular design lets details handle the fast local
decisions and gives a shared region a path to affect several details. In a [custom
connectome](anatomy.md), check both directions. The region needs a way to affect the motor
and a directed path back from a taught motor. An apical return can carry a correction
during a lesson, but it is not a substitute for the sensory or contextual drive a region
needs to operate.

Pass only the available observations and observable motor targets. Leave unobserved slots
[open](teach.md#a-slot-the-lesson-leaves-open). In a continuing task, let earlier observations
enter through `act` and teach the next decision before that moment is recorded, for example
with `step(teacher=...)`. Reset at independent stream boundaries, not at every decision.
For a small runnable example, see [the supervised teaching guide](teach.md#when-a-deep-brains-motor-learns-but-its-hidden-regions-do-not).

Route learned code and recall into the regions that have to interpret them. A retrieved
value copied directly into a motor can solve a repeat-last-output task without the hidden
region doing any work. Use several heads only if the task benefits from several reads, and
test what each contributes. With recall to a local plan, v0.104.0 reads the actual receiving
neurons when learning that head's address, rather than assuming it writes to `association`.

## Read the lesson before changing the gain

Enable `learning_credit_window=15` when composing or with `retune`. Read `credit_*` in the
lesson report, or `demonstration_credit_*` in `brain.last_learning` after `step`. Check that
there were actually taught slots and nonzero motor contrast before comparing regions.

| What the report shows | What to check next |
| --- | --- |
| A region's `return_share` is zero or partial | Check the directed motor return and whether the intended units are connected. A gain setting cannot create missing contacts. |
| Motor moves but hidden voltage contrast is tiny | Compare `max_return_hops` with the signed lesson budget. Inspect attenuation and the feedback route before widening the brain. |
| Hidden voltage changes but activation barely moves and `slope` is small | Look for saturation from standing motor return, basal sensory or memory drive, or bias. Changing the operating point can help more than increasing the correction. |
| Hidden activation changes but held-out behavior does not | Check for a motor shortcut, an ambiguous target, missing earlier context, or a task a local branch can already solve. An activity contrast is not evidence of a learned function. |

The warning refers to observed nonzero motor corrections. It fires once per streak when
a measured population stays at or below 0.1% of motor activation contrast for the chosen window.
It does not stop training or measure what a region learned. No warning does not prove a
healthy graph. Record both absolute contrast and its ratio to motor contrast. A large
multiple of an almost-zero response may still be unusable.

In v0.104.0 a finite lesson spends its configured signed sweep budget when the per-slot
pull is too small for the coarse movement tolerance. Qualified lessons instead bound each
signed tolerance by the actual nonzero force and reject unresolved motor potentials before
updating. Neither repair raises `beta`, connects absent returns or promises sufficient
credit beyond the sweep budget. Re-run your old failure before changing settings.
[Teaching phases and diagnostics](settling.md#measuring-credit-and-limiting-taught-drive)

## Try one intervention at a time

- **Saturation**. Compare ordinary teaching with an experimental `learning_drive_budget`,
  or try bounded, label-free `learning_credit_calibration` on a composed brain. Calibration
  checks existing return paths and preserves greedy motor choices on its probe rows. It
  leaves the circuit unchanged if it cannot find a safe improvement on those rows.
  [Drive and calibration](settling.md#automatic-deep-credit-calibration)
- **Weak apical correction**. Compare `learning_credit_transport=3.0` with `1.0` and `0.0`.
  Factor one controls for the more precise paired solve without amplification. This mode
  amplifies the change from a free reference in teaching only. Check the admitted factor
  and extra work, not just the requested factor. It does not change ordinary act or reward
  eligibility. [Teaching-only transport](settling.md#transporting-a-correction-without-amplifying-ordinary-return)
- **Recall favors high-norm recent keys**. For a fresh brain, compare the opt-in
  `recall_key_norm=True` with ordinary matching. This normalizes nonzero stored keys, not
  stored values, and is deliberately separate from `recall_cortex_norm`. It is a saved
  choice of geometry. Do not retrofit it by editing a checkpoint or combine it with
  `recall_action`. [Recall addressing and learning](memory.md#how-it-writes-and-how-the-address-learns)

These are experiments, not settings every deep brain needs. Keep the same examples, seeds,
and motor force when comparing. Recheck both behavior and cost on the deployment backend.
Custom recall readback can use the host even when the main settle uses an accelerator.

## Prove that the hidden region helps

Use held-out decisions where local and recent observations match but the right answer
depends on earlier context or on another branch. Compare the intact brain with a shallow
or local baseline, a matched run with the intended hidden plasticity frozen, and a private
evaluation copy with the coordinator's outgoing influence muted. Keep the coordinator's
own motor slots and local action paths intact in the last test. A drop under muting only
shows the circuit matters at play time. Acquisition over a frozen training control is the
separate check that it learned the function.

If the claim is learned recall, also replay with the learned address maps reset to their
initial values and do a targeted older-record swap, holding recent state fixed where
possible. The bank can teach a distinction that recent neural state then carries. Clearing
the bank at the decision, watching a head's weights move, or inspecting its attention alone
cannot distinguish those cases. Report the intact and control results over multiple seeds
before promoting an architecture. [Test memory relevance](evaluate.md#test-learned-memory-relevance)
