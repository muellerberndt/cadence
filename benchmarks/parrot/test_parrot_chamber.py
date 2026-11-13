"""The parrot chamber's own tests: the matched controls (E5) and the turn-taking rung (T0)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("torch")


@pytest.mark.parametrize("kind", ["cnn", "vit"])
def test_a_control_takes_the_reach_place(kind):
    from benchmarks.parrot.controls import ImageReach
    from cadence.parrot import Scene, Thing, render
    from cadence.parrot.brain import Parrot

    size = Parrot(seed=0).distance.parameters()
    control = ImageReach(kind, size, seed=0)
    assert abs(control.parameters() - size) <= 0.2 * size  # matched parameters
    assert control.macs() > 0
    parrot = Parrot(seed=0, control=control)
    scene = Scene([Thing("a", "apple", 0.03, 0.33, stand=0.12),
                   Thing("b", "cup", 0.3, 0.1, stand=0.05)])
    for _ in range(3):
        out = parrot.moment(render(scene, parrot.body.eyes()).images)
    both = [s for s in out["sightings"]]
    assert both and all(np.isfinite(s.distance) for s in both)
    query = both[0].query
    before = control.read([query])[0][0]
    for _ in range(25):
        control.teach([query], np.array([0.9]))
    after = control.read([query])[0][0]
    assert abs(np.log(after / 0.9)) < abs(np.log(before / 0.9))  # it learns from its lessons
