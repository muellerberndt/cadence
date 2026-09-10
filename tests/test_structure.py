from __future__ import annotations

import numpy as np

import cadence as cd
from cadence.structure import Seams, SleepConfig


def _learner(seed: int = 0) -> cd.Learner:
    wiring = cd.layered(6, 8, 2, density=1.0, seed=seed)
    return cd.Learner(cd.Settlement(wiring, cd.learning_rule()), wiring.sets["output"], cd.LearnerConfig(eta=1.0))


def test_sleep_consolidates_tagged_fast_strength_and_the_learner_only_moves_live_seams() -> None:
    learner = _learner()
    seams = Seams(learner, SleepConfig(consolidate=0.5, downscale=0.5, tag_saturation=0.01))
    assert seams.slow_fraction() == 0.0
    rng = np.random.default_rng(0)
    x = rng.random((16, 6))
    drive = learner.engine.clamp_levels(np.pad(x, ((0, 0), (0, learner.engine.wiring.n - 6))))
    y = (x[:, 0] > 0.5).astype(int)
    for _ in range(5):
        learner.step(drive, y)
        seams.observe()
    tagged = seams.tag >= 0.01
    fast_before = seams.fast.copy()
    report = seams.sleep()
    assert report["day"] == 1
    assert seams.slow_fraction() > 0.0
    # a fully tagged seam is consolidated: its slow part is half of what its fast part was
    assert np.allclose(seams.slow[tagged], 0.5 * fast_before[tagged])
    # the learner's trainable mask equals alive, so a pruned overlap never moves again
    seams.alive[0] = False
    seams._enforce()
    assert learner.trainable_overlaps[0] == False  # noqa: E712
    learner.step(drive, y)
    assert learner.engine.edge_scale[0] == 0.0


def test_prune_and_sprout_change_the_wiring_digest_and_respect_the_budget() -> None:
    learner = _learner(1)
    w = learner.engine.wiring
    seams = Seams(learner, SleepConfig(prune_below=0.05, sprout_above=0.1, budget=3))
    digest0 = seams.digest()
    # make half the seams weak: they are pruned in sleep
    weak = np.arange(w.edges) % 2 == 0
    seams.fast[weak] = 0.001
    seams._enforce()
    expected = int((np.abs(seams.slow + 0.5 * seams.fast) < 0.05).sum())  # untagged fast strength is halved before the prune; lateral overlaps start at zero
    report = seams.sleep()
    assert report["pruned"] == expected
    assert seams.digest() != digest0
    # co-activation on silent overlaps makes them sprout, within each owner's budget
    activation = np.ones((4, w.n))
    seams.observe(activation)
    report = seams.sleep()
    assert report["sprouted"] > 0
    in_degree = np.bincount(w.post[seams.alive], minlength=w.n)
    assert in_degree.max() <= max(3, np.bincount(w.post[~weak], minlength=w.n).max())


def test_conserved_strength_scales_an_owner_s_incoming_seams() -> None:
    learner = _learner(2)
    seams = Seams(learner, SleepConfig(strength=1.0))
    w = learner.engine.wiring
    magnitude = np.abs(learner.engine.edge_scale)
    total = np.bincount(w.post, weights=magnitude, minlength=w.n)
    assert total.max() <= 1.0 + 1e-9
