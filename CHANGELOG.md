# Changelog

## 0.7.0 (2026-09-11)

The accelerators, in the core.

- `backend="mlx"`: the settlement on Apple silicon through MLX (`pip install
  "cadence-net[apple]"`): the block transport as device matrix products in float32 on the
  unified memory, nudges, adaptation, masks, tolerance, trajectories and `repair` as on the
  other backends; checked against the CPU engine to 1e-4 with every feature.
- A device handle on `SettledState` (`state.device`): a settlement that continues from a
  state the same engine produced starts from the device copy instead of uploading the host
  arrays, and `Settlement.contrast_on_device` gives the learning rule its per-overlap
  contrast as block Gram products on the device, so only one number per overlap comes back.
  `Learner.contrast` uses it when both phases carry the handle; on torch/CUDA float64 the
  learned scales agree with the CPU engine to 2e-16, on MPS and MLX float32 to 2e-7.
- `available_backends()` lists `mlx`; the backends doc has a hardware guide (Apple silicon,
  Intel and AMD CPUs, NVIDIA) with measured updates at language-model shapes.

## 0.6.0 (2026-09-11)

The cost of a settlement step is now the cost of the owners that move.

- `cadence.blocks`: the block transport. The overlap matrix is stored as dense blocks
  between the contiguous ranges the wiring's named sets cut, and the product of a range
  whose activation did not change since the previous step is reused. The clamped inputs of
  a layered net are multiplied once per settlement instead of once per step; the inbox is
  the same sum in a different association order, and the conformance check against the
  owner-by-owner reference still holds to rounding (2e-16). `Settlement(..., layout=)` and
  `Settlement.layout`; `to_dict()` reports the layout. The `dense_limit` now bounds the
  block entries, so a layered net of several thousand owners settles on blocks.
- The fused kernel skips owners whose potential did not move and freezes ranges that hear
  nothing once they are still (exact: the update of such an owner is a fixed function of
  its own state), and no longer recomputes the activation of a state it continues from.
- The learning rule's contrast is one small Gram product per block, and one product
  instead of two for a range that is the same in both phases.
- On the MNIST shape (784 inputs, 256 hidden, batch 256) one learning update went from
  63 ms to 18 ms on one M4 core, and from 39 ms to 7 ms at 32 hidden owners; before, the
  cost of an update barely depended on the hidden width because the input block dominated.
- `cadence.timing`: `latency(decide)` times one decision many times and reports the median,
  the tails, and the scheduler's context switches from `getrusage`; `environment()` records
  threads, pinning (Linux), load and library versions for a receipt.
- `Settlement(precision=)` on the torch backend: float32 on CUDA when speed matters more than
  the receipt (read out on the cpu backend), float64 where the device has it.
- `cadence.stream`: owned state as a clamp. `stateful(...)` builds a windowed net with a range
  of context owners, one per hidden owner, wired densely into the hidden owners; `Echo` keeps
  a leaky trace of the hidden owners' equilibria across a batch of streams and writes it into
  the context clamp, so each state reverberates and fades. A test learns to name the symbol
  seen one input ago from a window of one, which no window can.
- `SettledState.repair`: the total movement of the published activations during a settlement,
  one number per row, on every backend. It is the work the net did to get from where it was
  to rest: the surprise of an input, measured rather than inferred from the step count.
- `Learner(slots=)`: the output owners as equal groups, each its own softmax choice, all nudged
  together; `targets` takes one label per slot and `predict` returns one choice per slot. A
  whole utterance settles at once. Each slot's nudge carries `beta / slots`, so the contrast
  is the gradient of the mean loss over the slots and `eta` means the same at any count
  (a span of four at full beta drove the hidden owners into saturation).
- `cadence.constitution`: where a wiring comes from. `Constitution` (regions and projections as
  a few numbers each), `grow` (development into a wiring with named sets, deterministic in
  the seed), `mutate`, `evolve` (selection over constitutions under a fitness the caller
  supplies). The phase before learning; not a new primitive.
- CI is green again: the 0.5.0 modules are formatted and typed.
- Tests: 58.

## 0.5.0 (2026-09-10)

Learning from reward and a life for the seams, built for the paper "You don't need attention
after all" and measured on its gates.

- `ActorCritic` (`cadence.plasticity`): the three-factor rule. An eligibility trace at every
  seam of the free/nudged contrast for the action taken, a linear critic on named owners
  with its own trace, a dopamine owner broadcasting the temporal-difference error; the
  adaptive local step (`momentum`, `normalize`, bias-corrected); `dopamine_cap`,
  `dopamine_center`, `critic_normalize`; `bootstrap=` for time limits. `Population` for
  continuous actions as a bump code. Cart-pole: 500 on every seed with the threshold at 40k
  to 60k steps, 1,716 parameters, two warm settlement steps per decision at deployment.
- `DreamActorCritic` and `actor_critic_wiring` (`cadence.dream`): an actor and a critic in
  one net with a memory; imagine-and-feel action selection (candidates felt in the critic),
  the critic's dream toward a target read with the slow strengths, the actor imitating the
  action taken. `DiscreteCode` for discrete actions.
- `Seams` and `SleepConfig` (`cadence.structure`): fast and slow strengths, tags, sleep
  (consolidate, downscale, prune, sprout within a budget), conserved incoming strength.
- `cadence.fused`: a compiled dense settlement kernel and a fused three-factor step on the
  CPU backend when numba is installed (`pip install "cadence-net[fast]"`); identical
  arithmetic, checked against the NumPy loop to 2e-16; `CADENCE_FUSED=0` forces the loop.
- `Learner.apply` and `Learner.contrast_rows`; the contrast as a Gram matrix product read
  at the overlaps (forty times faster than the gather on dense wirings).
- Docs: `reward.md`, `life.md`; the capability table carries the gates.
- Tests: 50.

## 0.4.1 (2026-09-09)

- Docs only. The grey parrot rung was withdrawn from cadence-examples (its imitations did not
  reach the bar); `embodied.md` now works through cart-pole and the sign writer, `tasks.md`
  keeps the several-learners-in-one-net recipe without the parrot, and the examples table
  lists the nine rungs.

## 0.4.0 (2026-09-09)

- `Learner.save` / `Learner.load` (`cadence.save`, `cadence.load`): one-file checkpoints of a
  trained learner, wiring and parameters included, that load on any backend and keep learning.
- `LearnerConfig.decay`: a leak on the seams, so a net that never stops learning stays plastic.
- `LearnerConfig.momentum`: each seam steps on a running average of its own contrast.
- `Learner(trainable_owners=...)` beside `trainable_overlaps`: two learners can share one net,
  each moving and decaying only its own seams and owners; a frozen overlap never moves, not
  even through the seam tying.
- `Learner(tie_groups=...)` and `embedded(...)`: a shared embedding across window positions.
- `py.typed`: the package is typed; `mypy --strict` clean.
- Tests: 42 across every module, 96% line coverage; the torch kernel is checked against the
  CPU engine with nudges, weights, adaptation and trajectories.
- Docs: task recipes (tabular place codes, regression as a pattern, streams, few labels, several
  learners in one net), `embodied.md` for deployment.

## 0.3.0

- The free/nudged learning rule, layered wirings, the torch backend, receipts, the examples ladder.
