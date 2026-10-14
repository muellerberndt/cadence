"""Post-hoc (declared after the confirmation run): which 0.003-tolerance trajectory
stays closer to a tightly solved one?

Repeats the composed_default 24-lesson trajectories of run.py at tolerance 1e-8
(damping 6, budget 65536, default schedule) and compares free answers and
efficacies with the default and candidate arms of the confirmation receipt.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from run import FIXTURE, HERE, compose

receipt = json.loads(Path(sys.argv[1]).read_text())
protocol = receipt["protocol"]
case = protocol["cases"]["composed_default"]
with np.load(FIXTURE, allow_pickle=False) as arrays:
    inputs, labels = arrays["school_inputs"], arrays["school_labels"]
out = []
for seed, founder in zip(receipt["seeds"], receipt["cases"]["composed_default"]["trajectory"],
                         strict=True):
    brain = compose(case, seed)
    brain.learner.config = replace(brain.learner.config, tolerance=1e-8, damping=6,
                                   free_steps=65536, nudged_steps=65536)
    rng = np.random.default_rng(2000 + seed)
    for _ in range(protocol["trajectory"]):
        rows = rng.choice(len(labels), protocol["batch"], replace=False)
        brain.learner.step(brain.stimulus(inputs[rows], memory=False), labels[rows])
    brain.learner.config = replace(brain.learner.config, tolerance=case["tolerance"], damping=3,
                                   free_steps=case["budget"], nudged_steps=case["budget"])
    tight = np.array(brain.learner.predict(brain.stimulus(inputs, memory=False)))
    row = {"seed": seed, "tight_recall": float((tight == labels).mean())}
    for arm in ("default", "candidate"):
        answers = np.array(founder[arm]["answers"])
        row[f"{arm}_answers_matching_tight"] = int((answers == tight).sum())
    out.append(row)
    print(row, flush=True)
(HERE / "results" / "posthoc-tight-2026-10-08.json").write_text(json.dumps(out, indent=1))
