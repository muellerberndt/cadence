"""A patch net learns to tell two noisy patterns apart with the free/nudged rule.

Run:  python examples/two_blobs.py
"""

from __future__ import annotations

import numpy as np

import cadence as cd

rng = np.random.default_rng(0)
x = np.zeros((120, 8))
y = np.repeat([0, 1], 60)
x[:60, :4] = 1.0  # class 0 lights the left half, class 1 the right half
x[60:, 4:] = 1.0
x = np.clip(x + 0.3 * rng.standard_normal(x.shape), 0.0, 1.0)

wiring = cd.layered(8, 16, 2, density=0.6, seed=1)  # sets: input, hidden, output
learner = cd.Learner(
    cd.Settlement(wiring, cd.learning_rule()), wiring.sets["output"], cd.LearnerConfig(eta=2.0)
)
drive = learner.engine.clamp_levels(np.pad(x, ((0, 0), (0, wiring.n - 8))))

print(f"before learning: {learner.accuracy(drive, y):.2f}")
for epoch in range(3):
    order = rng.permutation(len(y))
    for start in range(0, len(y), 20):
        idx = order[start : start + 20]
        _, report = learner.step(drive[idx], y[idx])
    print(
        f"epoch {epoch + 1}: accuracy {learner.accuracy(drive, y):.2f}, "
        f"free phase {report['free_steps']:.0f} steps, nudged {report['nudged_steps']:.0f}"
    )
print(f"{learner.parameters()} parameters; the free phase never saw a label")
