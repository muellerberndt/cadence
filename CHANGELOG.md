# Changelog

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
