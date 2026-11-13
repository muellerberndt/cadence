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


def test_t0_the_onset_learns_the_gap_and_the_yoked_brain_does_not():
    """Rung T0 on a tiny trainer: one founder's voice learns to reply in the gap after the
    trainer speaks; a yoked brain paid at the same moments, whatever it does, learns no turns."""
    from benchmarks.parrot.turns import exchanges, readings
    from cadence.parrot.voice import Voice

    life = exchanges(Voice(seed=0), np.random.default_rng(0), 150)
    yoked = exchanges(Voice(seed=100), np.random.default_rng(100), 150,
                      yoked=set(life["paid"]))
    first, last = readings(life, 0), readings(life, 100)
    assert last["in_window"] >= 0.5 and last["in_window"] > first["in_window"] + 0.2
    assert last["overlap"] < first["overlap"]
    assert readings(yoked, 100)["in_window"] <= 0.2


def test_the_voice_opens_the_beak():
    from cadence.parrot import Scene, Thing, render
    from cadence.parrot.brain import Parrot
    from cadence.parrot.voice import Voice
    from cadence.parrot.world import NAMES

    parrot = Parrot(seed=0, voice=Voice(seed=1, temperature=5.0))
    scene = Scene([Thing("a", "apple", 0.03, 0.33, stand=0.12)])
    opened = []
    for _ in range(6):
        out = parrot.moment(render(scene, parrot.body.eyes()).images, reward=0.0)
        opened.append((out["vocalized"], parrot.body.targets[NAMES.index("beak_open")]))
    assert any(v and t > 0 for v, t in opened) or all(not v for v, _ in opened)
    assert all((t > 0) == v for v, t in opened)
