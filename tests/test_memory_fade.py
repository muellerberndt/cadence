"""``memory_fade``: records nobody uses fade, slower the deeper their use (#253).

Each key coordinate of the store's consolidated records keeps a depth (the unit cue's magnitude
summed over the moments that cue it and the writes that use it) and a trace of its recent
cueing. Every moment of life an idle record keeps ``1 - fade / (1 + (depth / FADE_SCALE) **
FADE_ORDER)`` of itself. Off (the founder) the brain is the founder's to the bit.
"""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import numpy as np
import pytest

import cadence as cd
from cadence.memory import SynapticMemory


def life(brain, moments=300, seed=1, idle=0):
    """Six binary inputs, three actions, paid when the action is the count of active inputs
    modulo three; then ``idle`` blank moments with nothing paid."""
    rng = np.random.default_rng(seed)
    reward = None
    for _ in range(moments):
        x = (rng.random((1, 6)) < 0.4).astype(float)
        action = brain.live(x, reward=None if reward is None else np.array([reward]))
        reward = float(action[0] == int(x[0].sum()) % 3)
    for _ in range(idle):
        brain.live(np.zeros((1, 6)), reward=None if reward is None else np.array([reward]))
        reward = 0.0
    return brain


def checkpoint(brain, folder):
    path = Path(folder) / "brain.npz"
    brain.save(path)
    return path


def test_off_is_the_founder_to_the_bit():
    founder = life(cd.Brain.compose(6, 3, seed=2, arousal=True, recall=False))
    off = life(cd.Brain.compose(6, 3, seed=2, arousal=True, memory_fade=0.0, recall=False))
    assert off.hippocampus.fade == 0.0 and off.hippocampus.fade_depth is None
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        left, right = np.load(checkpoint(founder, a)), np.load(checkpoint(off, b))
        assert sorted(left.files) == sorted(right.files)
        for name in left.files:
            assert np.array_equal(left[name], right[name]), name
    assert "memory_fade" not in founder.describe()["genes"]
    assert founder.parameters() == off.parameters()


def test_a_record_in_use_keeps_and_an_idle_one_fades():
    store = SynapticMemory(np.arange(4), np.arange(4, 6), fade=0.01)
    key, value = np.eye(4)[[0]], np.array([[1.0, 0.0]])
    store.reset(1)
    for _ in range(50):  # coordinate 0 written and cued, coordinate 1 written once then idle
        store.observe(key, value)
        store.fade_moment(key)
    store.observe(np.eye(4)[[1]], value)
    used = store.consolidated.copy()
    for _ in range(200):  # coordinate 0 stays in use
        store.fade_moment(key)
    assert store.consolidated[0, 0] == pytest.approx(used[0, 0], rel=1e-6)
    assert store.consolidated[1, 0] < 0.5 * used[1, 0]
    assert store.fade_depth[0] > store.fade_depth[1] > 0


def test_a_deep_record_fades_far_more_slowly():
    shallow = SynapticMemory(np.arange(2), np.arange(2, 4), fade=1e-3)
    deep = SynapticMemory(np.arange(2), np.arange(2, 4), fade=1e-3)
    key, value = np.eye(2)[[0]], np.array([[1.0, 0.0]])
    for store, moments in ((shallow, 100), (deep, 6000)):
        store.reset(1)
        for _ in range(moments):
            store.observe(key, value)
            store.fade_moment(key)
    start = shallow.consolidated[0, 0], deep.consolidated[0, 0]
    for _ in range(30000):  # idle: nothing cues either
        shallow.fade_moment(None)
        deep.fade_moment(None)
    assert shallow.consolidated[0, 0] < 0.05 * start[0]
    assert deep.consolidated[0, 0] > 0.9 * start[1]


def test_every_moment_of_life_cues_and_fades():
    brain = life(cd.Brain.compose(6, 3, seed=2, arousal=True, memory_fade=0.05, recall=False),
                 moments=200)
    store = brain.hippocampus
    depth, recent = store.fade_depth.copy(), store.fade_recent.copy()
    assert depth.min() > 0 and recent.min() > 0
    life(brain, moments=0, idle=100)  # blank moments cue nothing: the depth stays, the trace decays
    assert np.array_equal(store.fade_depth, depth)
    assert np.allclose(store.fade_recent, recent * (1 - 1 / store.FADE_IDLE) ** 100)
    store.FADE_IDLE = 5.0  # a short memory of use: the idle records fade within the stretch
    before = np.abs(store.consolidated).sum()
    life(brain, moments=0, idle=400)
    assert np.abs(store.consolidated).sum() < 0.5 * before


def test_save_and_load_continue_bit_for_bit():
    brain = life(cd.Brain.compose(6, 3, seed=2, arousal=True, memory_fade=0.01, maturity=True,
                                  recall=False), idle=50)
    with tempfile.TemporaryDirectory() as folder:
        loaded = cd.Brain.load(checkpoint(brain, folder))
    for name in ("fade_depth", "fade_recent", "consolidated"):
        assert np.array_equal(getattr(loaded.hippocampus, name), getattr(brain.hippocampus, name))
    assert loaded.hippocampus.fade == brain.hippocampus.fade
    assert loaded.describe()["genes"]["memory_fade"] == 0.01
    life(brain, moments=60, seed=4, idle=20)
    life(loaded, moments=60, seed=4, idle=20)
    for name in ("fade_depth", "fade_recent", "consolidated"):
        assert np.array_equal(getattr(loaded.hippocampus, name), getattr(brain.hippocampus, name))


def test_parameters_count_the_depths_and_traces():
    off = cd.Brain.compose(6, 3, seed=2, recall=False)
    on = cd.Brain.compose(6, 3, seed=2, memory_fade=0.01, recall=False)
    assert on.parameters() - off.parameters() == 2 * on.hippocampus.key_width


def test_refusals():
    with pytest.raises(ValueError, match="memory_fade must lie in"):
        cd.Brain.compose(6, 3, memory_fade=1.0, recall=False)
    with pytest.raises(ValueError, match="memory_fade"):
        cd.Brain.compose(6, 3, memory_fade=-0.1, recall=False)
    with pytest.raises(ValueError, match="memory_fade needs the episodic memory"):
        cd.Brain.compose(6, 3, memory_fade=0.01, episodic=False, recall=False)
    brain = cd.Brain.compose(6, 3, memory_fade=0.01, recall=False)
    with pytest.raises(ValueError, match="construction"):
        brain.retune(memory_fade=0.02)


def test_step_moments_fade_and_a_refused_moment_changes_nothing(monkeypatch):
    brain = cd.Brain.compose(6, 3, seed=2, memory_fade=0.01, recall=False)
    brain.step(np.ones((1, 6)))
    brain.step(np.ones((1, 6)), reward=np.array([1.0]))
    assert brain.hippocampus.fade_depth.sum() > 0
    before = {n: getattr(brain.hippocampus, n).copy() for n in ("fade_depth", "fade_recent", "consolidated")}
    with pytest.raises(ValueError):
        brain.step(np.ones((1, 5)))  # the wrong width is refused before the moment
    for name, value in before.items():
        assert np.array_equal(getattr(brain.hippocampus, name), value), name


def test_learn_on_its_own_cues_by_the_next_observations():
    brain = cd.Brain.compose(6, 3, seed=2, memory_fade=0.01, recall=False)
    x = np.eye(6)[[1]]
    brain.act(x)
    depth = brain.hippocampus.fade_depth.copy()
    recent = brain.hippocampus.fade_recent.copy()
    brain.learn(np.array([1.0]), np.array([False]), np.eye(6)[[4]])  # reward, done, next
    kept = 1 - 1 / brain.hippocampus.FADE_IDLE  # the trace takes the cue, then decays a moment
    moved = brain.hippocampus.fade_recent - recent * kept
    assert moved[4] == pytest.approx(kept) and np.allclose(moved[[0, 1, 2, 3, 5]], 0.0)
    assert brain.hippocampus.fade_depth[4] - depth[4] == pytest.approx(1.0)
    one = cd.Brain.compose(1, 2, seed=2, memory_fade=0.01, recall=False)  # a key as wide as the reward
    one.act(np.zeros((1, 1)))
    one.learn(np.array([1.0]), np.array([False]), np.zeros((1, 1)))  # a blank moment cues nothing
    assert not one.hippocampus.fade_recent.any()


def test_grow_and_graft_keep_the_fade():
    brain = life(cd.Brain.compose(6, 3, seed=2, arousal=True, memory_fade=0.01, recall=False),
                 moments=120, idle=30)
    grown = brain.grow(efference_amplitude=1.0)
    assert grown.hippocampus.fade == brain.hippocampus.fade
    for name in ("fade_depth", "fade_recent"):
        assert np.array_equal(getattr(grown.hippocampus, name), getattr(brain.hippocampus, name))
    grafted = brain.graft(copy.deepcopy(brain), ["sensory", "association", "motor"])
    assert grafted.hippocampus.fade == brain.hippocampus.fade
    for name in ("fade_depth", "fade_recent"):
        assert np.array_equal(getattr(grafted.hippocampus, name), getattr(brain.hippocampus, name))


def test_clear_erases_the_depths_with_the_records():
    brain = life(cd.Brain.compose(6, 3, seed=2, arousal=True, memory_fade=0.01, recall=False),
                 moments=80)
    brain.hippocampus.clear()
    assert not brain.hippocampus.fade_depth.any() and not brain.hippocampus.fade_recent.any()
