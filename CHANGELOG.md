# Changelog

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
- Docs: task recipes (tabular place codes, regression as a pattern, streams, few labels, vocal
  learning), `embodied.md` for deployment.

## 0.3.0

- The free/nudged learning rule, layered wirings, the torch backend, receipts, the examples ladder.
