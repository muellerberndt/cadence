"""One free/nudged update on a six-owner net, every number printed (docs/learning.md, section 4).

Run:  python examples/worked_update.py
"""

from __future__ import annotations

import numpy as np

import cadence as cd

wiring = cd.layered(2, 2, 2, density=1.0, seed=3)
config = cd.LearnerConfig(
    eta=1.0, beta=0.1, temperature=0.2, tolerance=1e-9, free_steps=500, nudged_steps=500
)
learner = cd.Learner(cd.Settlement(wiring, cd.learning_rule(dt=1.0)), wiring.sets["output"], config)
w = wiring
print("owners: inputs", w.sets["input"], "hidden", w.sets["hidden"], "outputs", w.sets["output"])
print("seams (pre -> post: scale):")
for e in range(w.edges):
    print(f"  {w.pre[e]} -> {w.post[e]}: {learner.engine.edge_scale[e]:+.3f}")

drive = learner.engine.clamp_levels(np.pad([[1.0, 0.2]], ((0, 0), (0, w.n - 2))))
label = np.array([1])
free = learner.free(drive)
print(f"\nfree phase, {free.steps} steps:")
print(f"  v = {np.round(free.v[0], 3)}\n  s = {np.round(free.activation[0], 3)}")
out = free.activation[0, learner.output_index]
p = np.exp(out / config.temperature) / np.exp(out / config.temperature).sum()
print(f"outputs {np.round(out, 3)}, softmax(s/T) {np.round(p, 3)}")
print(f"answer {out.argmax()} (label {label[0]})")

target = learner.targets(label)
plus = learner.nudged(drive, free, target)
minus = learner.nudged(drive, free, target, sign=-1.0)
first_push = config.beta * (target[0, learner.output_index] - p)
print(f"nudge drive at the first +beta step on the outputs: {np.round(first_push, 3)}")
print(f"+beta rest, {plus.steps} steps:  s = {np.round(plus.activation[0], 3)}")
print(f"-beta rest, {minus.steps} steps:  s = {np.round(minus.activation[0], 3)}")

overlap_term, owner_term = learner.contrast(free, plus, minus)
before = learner.engine.edge_scale.copy()
learner.update(free, plus, minus)
print("\ncontrast per overlap and the resulting scale (eta 1, tied pairs averaged):")
for e in range(w.edges):
    now = learner.engine.edge_scale[e]
    seam = f"{w.pre[e]} -> {w.post[e]}"
    print(f"  {seam}: contrast {overlap_term[e]:+.4f}   scale {before[e]:+.3f} -> {now:+.4f}")
print("bias change per owner:", np.round(config.eta_bias * owner_term, 5))
after = learner.free(drive).activation[0, learner.output_index]
print(f"\noutputs after one update: {np.round(after, 3)}   (before: {np.round(out, 3)})")
