"""``maturity``: one slow variable per brain, the use of its plasticity (#253).

Each actor update adds its applied magnitude to the brain's use; every moment of life keeps
``1 - 1 / maturity_recovery`` of it. The step factor ``m = 1 / (1 + (use / maturity_scale) **
4)`` multiplies the actor's steps and engages the store's surprise-gated writes with strength
``1 - m``. Off (the founder) the brain is the founder's to the bit.
"""

from __future__ import annotations

import copy
import json
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pytest

import cadence as cd
from cadence.memory import SynapticMemory
from cadence.plasticity import Maturity


def life(brain, moments=300, seed=1):
    """A short continuing life: six binary inputs, three actions, the action paid when it is
    the count of active inputs modulo three."""
    rng = np.random.default_rng(seed)
    reward = None
    for _ in range(moments):
        x = (rng.random((1, 6)) < 0.4).astype(float)
        action = brain.live(x, reward=None if reward is None else np.array([reward]))
        reward = float(action[0] == int(x[0].sum()) % 3)
    return brain


def checkpoint(brain, folder):
    path = Path(folder) / "brain.npz"
    brain.save(path)
    return path


def test_off_is_the_founder_to_the_bit():
    founder = life(cd.Brain.compose(6, 3, seed=2, arousal=True, recall=False))
    off = life(cd.Brain.compose(6, 3, seed=2, arousal=True, maturity=False, recall=False))
    assert founder.basal_ganglia.maturity is None and off.hippocampus.gate_noise is None
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        with np.load(checkpoint(founder, a)) as left, np.load(checkpoint(off, b)) as right:
            assert sorted(left.files) == sorted(right.files)
            for name in left.files:
                assert np.array_equal(left[name], right[name]), name
    assert "maturity" not in founder.describe()["genes"]


def test_use_grows_with_the_updates_and_the_step_falls():
    young = life(cd.Brain.compose(6, 3, seed=2, arousal=True, recall=False))
    mature = cd.Brain.compose(6, 3, seed=2, arousal=True, maturity=True, maturity_scale=0.02, recall=False)
    start = mature.brain.efficacy.copy()
    life(mature)
    state = mature.basal_ganglia.maturity
    assert state.use > 0.02 and state.factor() < 0.5
    moved = np.abs(mature.brain.efficacy - start).sum()
    assert moved < np.abs(young.brain.efficacy - start).sum()
    genes = mature.describe()["genes"]
    assert genes["maturity"] is True and genes["maturity_scale"] == 0.02
    assert genes["maturity_recovery"] == 200_000.0


def test_an_idle_brain_recovers_and_a_young_one_learns_as_the_founder():
    state = Maturity(scale=7.5, recovery=1_000.0, use=15.0)
    before = state.factor()
    for _ in range(3_000):
        state.relax()
    assert state.use == pytest.approx(15.0 * (1 - 1e-3) ** 3_000)
    assert before < 0.1 < 0.9 < state.factor()
    # a scale far beyond any use: m is 1, the steps and the store's writes are the founder's
    founder = life(cd.Brain.compose(6, 3, seed=2, arousal=True, recall=False))
    young = life(cd.Brain.compose(6, 3, seed=2, arousal=True, maturity=True, maturity_scale=1e12, recall=False))
    # (within rounding: the gated write sums the residual's change in another order)
    np.testing.assert_allclose(young.brain.efficacy, founder.brain.efficacy, rtol=0, atol=1e-9)
    np.testing.assert_allclose(young.hippocampus.consolidated, founder.hippocampus.consolidated,
                               rtol=0, atol=1e-9)


def test_a_mature_store_moves_a_record_only_by_its_surprise():
    def store(use):
        memory = SynapticMemory(np.arange(2), np.arange(2), consolidation=1.0, rate=1.0)
        memory.reset(1)
        memory.gate_noise = np.full(memory.consolidated.shape, 0.01)  # noise scale 0.1
        memory.maturity = Maturity(scale=1.0, recovery=1e9, use=use)
        return memory

    key = np.array([[1.0, 0.0]])
    for error, moves in ((0.15, False), (1.0, True)):
        mature = store(use=1e6)  # m near 0: the gate at full strength
        mature.observe(key, np.array([[error, 0.0]]), value_mask=np.array([[True, False]]))
        # the record moves by the part of its error beyond two noise scales (0.2)
        expected = max(0.0, error - 0.2)
        assert mature.consolidated[0, 0] == pytest.approx(expected, abs=1e-9)
        assert bool(mature.consolidated[0, 0] > 0) is moves
        young = store(use=0.0)  # m 1: the founder's write
        young.observe(key, np.array([[error, 0.0]]), value_mask=np.array([[True, False]]))
        assert young.consolidated[0, 0] == pytest.approx(error)
    # each write moves the record's noise toward its squared error
    assert young.gate_noise[0, 0] == pytest.approx(0.01 + 0.05 * (1.0 - 0.01))
    assert young.gate_noise[0, 1] == 0.01


def test_save_and_load_continue_bit_for_bit():
    brain = life(cd.Brain.compose(6, 3, seed=2, arousal=True, maturity=True, maturity_scale=0.05, recall=False))
    with tempfile.TemporaryDirectory() as folder:
        loaded = cd.Brain.load(checkpoint(brain, folder))
    assert loaded.hippocampus.maturity is loaded.basal_ganglia.maturity
    assert loaded.basal_ganglia.maturity == brain.basal_ganglia.maturity
    life(brain, 150, seed=5)
    life(loaded, 150, seed=5)
    assert np.array_equal(loaded.brain.efficacy, brain.brain.efficacy)
    assert np.array_equal(loaded.hippocampus.consolidated, brain.hippocampus.consolidated)
    assert np.array_equal(loaded.hippocampus.gate_noise, brain.hippocampus.gate_noise)
    assert loaded.basal_ganglia.maturity.use == brain.basal_ganglia.maturity.use


def test_parameters_count_the_use_and_the_noise_scales():
    brain = cd.Brain.compose(6, 3, seed=2, arousal=True, maturity=True, recall=False)
    founder = cd.Brain.compose(6, 3, seed=2, arousal=True, recall=False)
    assert brain.parameters() == founder.parameters() + 1 + brain.hippocampus.gate_noise.size


def test_retune_changes_the_constants_and_not_the_structure():
    brain = cd.Brain.compose(6, 3, seed=2, arousal=True, maturity=True, recall=False)
    brain.retune(maturity_scale=9.0, maturity_recovery=50_000.0)
    state = brain.basal_ganglia.maturity
    assert (state.scale, state.recovery) == (9.0, 50_000.0)
    assert brain.hippocampus.maturity is state
    with pytest.raises(ValueError, match="construction setting"):
        brain.retune(maturity=False)
    with pytest.raises(ValueError, match="lie in"):
        brain.retune(maturity_recovery=0.5)
    with pytest.raises(ValueError, match="requires a brain composed with maturity"):
        cd.Brain.compose(6, 3, seed=2, arousal=True, recall=False).retune(maturity_scale=9.0)
    with pytest.raises(ValueError, match="boolean"):
        cd.Brain.compose(6, 3, seed=2, maturity=1, recall=False)


def test_without_the_store_the_maturity_scales_the_actor_alone():
    brain = life(cd.Brain.compose(6, 3, seed=2, arousal=True, episodic=False, maturity=True,
                                  maturity_scale=0.05, recall=False))
    assert brain.hippocampus is None and brain.basal_ganglia.maturity.use > 0


def test_step_and_learn_are_moments():
    brain = cd.Brain.compose(6, 3, seed=2, maturity=True, maturity_recovery=10.0, recall=False)
    brain.basal_ganglia.maturity.use = 1.0
    rng = np.random.default_rng(0)
    brain.step(rng.random((1, 6)), reward=None)  # one moment, no update: one relaxation
    assert brain.basal_ganglia.maturity.use == pytest.approx(0.9)


def test_grow_and_graft_keep_the_maturity():
    brain = life(cd.Brain.compose(6, 3, seed=2, arousal=True, maturity=True, maturity_scale=0.05, recall=False),
                 moments=200)
    grown = brain.grow(efference_amplitude=1.0)
    assert grown.basal_ganglia.maturity == brain.basal_ganglia.maturity
    assert grown.hippocampus.maturity is grown.basal_ganglia.maturity
    assert np.array_equal(grown.hippocampus.gate_noise, brain.hippocampus.gate_noise)
    donor = copy.deepcopy(brain)
    grafted = brain.graft(donor, ["sensory", "association", "motor"])
    assert grafted.basal_ganglia.maturity == brain.basal_ganglia.maturity
    assert np.array_equal(grafted.hippocampus.gate_noise, brain.hippocampus.gate_noise)


def stepped(brain, moments=40, seed=0):
    """A short life through ``step``: random binary inputs, a reward on half the moments."""
    rng = np.random.default_rng(seed)
    brain.step(np.ones((1, 6)))
    for _ in range(moments):
        brain.step((rng.random((1, 6)) < 0.4).astype(float),
                   reward=np.array([float(rng.random() < 0.5)]))
    return brain


def held(brain):
    """Everything a moment of a mature brain can change: the store, the actor, the use."""
    memory = brain.hippocampus
    return {
        **{name: np.array(getattr(memory, name), copy=True)
           for name in ("consolidated", "strength", "mass", "gate_noise")},
        "efficacy": np.array(brain.brain.efficacy, copy=True),
        "use": brain.basal_ganglia.maturity.use,
    }


def assert_held(brain, before):
    for name, value in held(brain).items():
        assert np.array_equal(value, before[name]), name


def test_a_refused_moment_changes_nothing(monkeypatch):
    brain = life(cd.Brain.compose(6, 3, seed=2, arousal=True, maturity=True, maturity_scale=0.05, recall=False),
                 moments=50)
    before = held(brain)
    with pytest.raises(ValueError, match="6 values per row"):
        brain.live(np.zeros((1, 7)))
    assert_held(brain, before)
    brain = stepped(cd.Brain.compose(6, 3, seed=2, maturity=True, maturity_scale=0.05, recall=False))
    before = held(brain)
    with pytest.raises(ValueError):
        brain.step(np.full((1, 6), np.nan))
    assert_held(brain, before)
    # a learn the actor refuses after the store's write: the store and its noise restored
    brain.step(np.ones((1, 6)))
    before = held(brain)
    twin = copy.deepcopy(brain)
    twin.learn(np.array([1.0]), np.array([False]), np.ones((1, 6)))
    assert not np.array_equal(twin.hippocampus.gate_noise, before["gate_noise"])

    def refused(*args, **kwargs):
        raise ValueError("refused update")

    monkeypatch.setattr(brain.basal_ganglia, "_learn", refused)
    with pytest.raises(ValueError, match="refused update"):
        brain.learn(np.array([1.0]), np.array([False]), np.ones((1, 6)))
    assert_held(brain, before)


def test_a_refused_store_write_keeps_the_noise_scales():
    memory = SynapticMemory(np.arange(2), np.arange(2), consolidation=1.0, rate=1.0)
    memory.consolidated = np.full((2, 2), 1.5e308)
    memory.reset(1)
    memory.gate_noise = np.full(memory.consolidated.shape, 0.01)
    memory.maturity = Maturity(scale=1.0, recovery=1e9, use=1e6)
    key, mask = np.array([[1.0, 0.0]]), np.array([[True, False]])
    before = {name: value.copy() for name, value in memory.arrays().items()}
    with pytest.raises(ValueError, match="overflowed"):
        memory.observe(key, np.array([[-1.5e308, 0.0]]), value_mask=mask)
    for name, value in memory.arrays().items():
        assert np.array_equal(value, before[name]), name


def test_a_vanishing_error_is_no_surprise_and_warns_nothing():
    memory = SynapticMemory(np.arange(2), np.arange(2), consolidation=1.0, rate=1.0)
    memory.reset(1)
    memory.gate_noise = np.ones(memory.consolidated.shape)  # noise scale 1
    memory.maturity = Maturity(scale=1.0, recovery=1e9, use=1e6)  # m near 0
    key, mask = np.array([[1.0, 0.0]]), np.array([[True, False]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the error's ratio to the noise leaves the float range
        memory.observe(key, np.array([[1e-310, 0.0]]), value_mask=mask)
    assert memory.consolidated[0, 0] == 0.0


def test_a_mature_brain_saves_a_format_of_its_own():
    mature = life(cd.Brain.compose(6, 3, seed=2, arousal=True, maturity=True, recall=False), moments=20)
    founder = life(cd.Brain.compose(6, 3, seed=2, arousal=True, recall=False), moments=20)
    with tempfile.TemporaryDirectory() as folder:
        path = checkpoint(founder, folder)
        assert json.loads(str(np.load(path)["brain"]))["format"] == "cadence-brain/1"
        path = checkpoint(mature, folder)
        arrays = dict(np.load(path))
        meta = json.loads(str(arrays["brain"]))
        assert meta["format"] == "cadence-brain/2"  # which 0.93.1 and earlier refuse
        for edit in ({"format": "cadence-brain/1"}, {"maturity": None}):
            changed = {**meta, **edit}
            if changed["maturity"] is None:
                changed.pop("maturity")
            np.savez(path, **{**arrays, "brain": np.array(json.dumps(changed))})
            with pytest.raises(ValueError, match="belongs to checkpoint format cadence-brain/2"):
                cd.Brain.load(path)


def test_several_writing_streams_keep_every_residual_around_the_shared_records():
    def store(gated):
        memory = SynapticMemory(np.arange(2), np.arange(2), consolidation=0.5, rate=1.0,
                                decay=1.0)
        memory.reset(3)
        if gated:
            memory.gate_noise = np.full(memory.consolidated.shape, 0.01)  # noise scale 0.1
            memory.maturity = Maturity(scale=1.0, recovery=1e9, use=1e6)  # m near 0
        # the third stream first gets a residual of its own
        memory.observe(np.array([[0, 0], [0, 0], [0.0, 1.0]]), np.array([[0, 0], [0, 0], [0.7, 0]]))
        return memory

    def residual(memory):
        return memory.strength - memory.consolidated

    key = np.array([[1.0, 0.0], [1.0, 0.0], [0, 0]])
    value = np.array([[0.05, 0], [1.0, 0], [0, 0]])  # the first stream confirmed, one surprised
    mask = np.array([[True, False], [True, False], [False, False]])
    changes = []
    for gated in (False, True):
        memory = store(gated)
        before = (memory.consolidated.copy(), residual(memory))
        memory.observe(key, value, value_mask=mask)
        changes.append((memory.consolidated - before[0], residual(memory) - before[1]))
    (shared, streams), (gated_shared, gated_streams) = changes
    # each row's gain: the confirmed stream's 0 (its error within two noise scales), the
    # surprised one's 0.8; the write's gain on the action is their mean
    gain = np.array([0.4, 1.0])
    np.testing.assert_allclose(gated_shared, gain * shared, rtol=0, atol=1e-12)
    np.testing.assert_allclose(gated_streams, gain * streams, rtol=0, atol=1e-12)
    assert np.abs(gated_streams[2]).max() < 1e-12  # the stream that did not write keeps its own


def test_an_outcome_whose_noise_overflows_is_refused_and_the_brain_still_loads():
    for arousal in (True, None):
        brain = cd.Brain.compose(6, 3, seed=2, arousal=arousal, maturity=True, recall=False)
        run = brain.live if arousal else brain.step
        run(np.ones((1, 6)))
        before = held(brain)
        # finite, but its square is not: refused before anything changes, no NaN written
        with pytest.raises(ValueError, match="overflowed"):
            run(np.ones((1, 6)), reward=np.array([1.3e155]))
        assert_held(brain, before)
        assert np.isfinite(brain.hippocampus.gate_noise).all()
        with tempfile.TemporaryDirectory() as folder:
            loaded = cd.Brain.load(checkpoint(brain, folder))
        assert np.array_equal(loaded.hippocampus.gate_noise, brain.hippocampus.gate_noise)


def test_the_scaled_steps_warn_only_at_construction():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        brain = cd.Brain.compose(6, 3, seed=2, arousal=True, maturity=True, maturity_scale=0.01,
                                 actor_eta=0.1, actor_eta_bias=0.5, recall=False)
        constructed = len(caught)
        life(brain, moments=30)
    assert constructed == 1 and len(caught) == 1  # eta_bias above eta, once
    assert brain.basal_ganglia.maturity.factor() < 1.0
    assert (brain.basal_ganglia.config.eta, brain.basal_ganglia.config.eta_bias) == (0.1, 0.5)


def test_the_maturity_settings_need_maturity_and_valid_values():
    for settings in ({"maturity_scale": -5.0}, {"maturity_recovery": float("nan")},
                     {"maturity_scale": 3.0}, {"maturity_recovery": 5_000.0}):
        with pytest.raises(ValueError, match="maturity"):
            cd.Brain.compose(6, 3, seed=2, **{"recall": False, **settings})
    with pytest.raises(ValueError, match="need maturity"):
        cd.Brain.compose(6, 3, seed=2, maturity_scale=3.0, recall=False)
    for settings in ({"maturity_scale": 0.0}, {"maturity_scale": float("inf")},
                     {"maturity_recovery": 0.5}, {"maturity_scale": True}):
        with pytest.raises(ValueError, match="maturity_"):
            cd.Brain.compose(6, 3, seed=2, maturity=True, **{"recall": False, **settings})
    # the founders, named, are the founder
    founder = cd.Brain.compose(6, 3, seed=2, maturity_scale=10.0, maturity_recovery=200_000.0, recall=False)
    assert founder.basal_ganglia.maturity is None


NOISE = "memories/hippocampus/gate_noise"


def tampered(brain, folder, edit):
    """The brain's checkpoint with its metadata and arrays edited, written back."""
    path = checkpoint(brain, folder)
    arrays = dict(np.load(path))
    meta = json.loads(str(arrays["brain"]))
    edit(meta, arrays)
    np.savez(path, **{**arrays, "brain": np.array(json.dumps(meta))})
    return path


def test_load_refuses_a_malformed_maturity_by_name():
    brain = life(cd.Brain.compose(6, 3, seed=2, arousal=True, maturity=True, maturity_scale=0.05, recall=False),
                 moments=30)

    def drop(name):
        return lambda meta, arrays: meta["maturity"].pop(name)

    def setting(name, value):
        return lambda meta, arrays: meta["maturity"].__setitem__(name, value)

    def gate(value):
        return lambda meta, arrays: meta["memories"]["hippocampus"].__setitem__(
            "surprise_gate", value
        )

    def noise(value):
        return lambda meta, arrays: arrays.__setitem__(NOISE, np.full_like(arrays[NOISE], value))

    def ungated(meta, arrays):
        meta["memories"]["hippocampus"].pop("surprise_gate")
        arrays.pop(NOISE)

    cases = [
        (drop("use"), "misses use"), (drop("scale"), "misses scale"),
        (drop("recovery"), "misses recovery"),
        (lambda meta, arrays: meta.__setitem__("maturity", [1, 2, 3]), "must map"),
        (lambda meta, arrays: meta.__setitem__("maturity", "on"), "must map"),
        (lambda meta, arrays: meta.__setitem__("maturity", None), "must map"),
        (setting("order", 2.0), "unknown saved maturity field: 'order'"),
        (setting("use", 1e300), "saved maturity use"), (setting("use", -1.0), "saved maturity use"),
        (setting("use", "1"), "saved maturity use"), (setting("scale", 0.0), "saved maturity scale"),
        (setting("recovery", 0.5), "saved maturity recovery"),
        (gate("yes"), "surprise_gate must be boolean"), (gate(1), "surprise_gate must be boolean"),
        (noise(-1.0), "gate_noise must be nonnegative"),
        (ungated, "requires the store's noise scales"),
    ]
    with tempfile.TemporaryDirectory() as folder:
        for edit, refusal in cases:
            with pytest.raises(ValueError, match=refusal):
                cd.Brain.load(tampered(brain, folder, edit))
        # noise scales saved as integers load as floating point and keep living
        loaded = cd.Brain.load(tampered(brain, folder, lambda meta, arrays: arrays.__setitem__(
            NOISE, np.zeros(arrays[NOISE].shape, dtype=np.int64))))
        assert loaded.hippocampus.gate_noise.dtype == np.float64
        life(loaded, moments=20)


def test_the_factor_reads_any_use():
    assert Maturity(scale=1e-12, recovery=10.0, use=1e70).factor() == 0.0
    assert Maturity(scale=1e-12, recovery=10.0, use=Maturity.LIMIT).factor() == pytest.approx(
        1e-304
    )
    assert Maturity(scale=1.0, recovery=10.0, use=2.0).factor() == 1.0 / 17.0


def test_the_torch_backend_measures_the_step_on_the_device(monkeypatch):
    pytest.importorskip("torch")
    settings = {"seed": 2, "arousal": True, "maturity": True, "maturity_scale": 0.05}
    brain = cd.Brain.compose(6, 3, backend="torch", device="cpu", **{"recall": False, **settings})
    kernel = type(brain.brain._torch)
    reads = []
    host_scale = kernel.host_scale

    def counted(self):
        reads.append(1)
        return host_scale(self)

    monkeypatch.setattr(kernel, "host_scale", counted)
    life(brain, moments=60)
    assert brain.basal_ganglia.updates > 0 and not reads  # no efficacy came to the host
    monkeypatch.undo()
    host = life(cd.Brain.compose(6, 3, **{"recall": False, **settings}), moments=60)
    assert brain.basal_ganglia.maturity.use == pytest.approx(host.basal_ganglia.maturity.use,
                                                             rel=1e-9)


def test_maturity_is_exported_and_gates_the_place_records_alone():
    assert cd.Maturity is Maturity and "Maturity" in cd.__all__
    brain = cd.Brain.compose(6, 3, seed=2, arousal=True, maturity=True, memory_choice=True,
                             memory_extinction=0.5, recall=False)
    assert brain.hippocampus.gate_noise is not None
    episodic = brain._episodic
    assert brain.context_memory is episodic.temporal and episodic.paid is not None
    for store in (episodic.temporal, episodic.paid):
        assert store.gate_noise is None and store.maturity is None
    # a checkpoint that gates the context records or the direct payments is refused
    brain = life(brain, moments=30)

    def gated(name):
        def edit(meta, arrays):
            meta["memories"]["episodic"][name]["surprise_gate"] = True
            prefix = f"memories/episodic/{name}/"
            arrays[prefix + "gate_noise"] = np.zeros_like(arrays[prefix + "consolidated"])
        return edit

    with tempfile.TemporaryDirectory() as folder:
        for name in ("context_memory", "paid_memory"):
            with pytest.raises(ValueError, match="only the place records are surprise-gated"):
                cd.Brain.load(tampered(brain, folder, gated(name)))
    # a brain at the limit stays there, at any scale, and saves a checkpoint that loads
    brain = cd.Brain.compose(6, 3, seed=2, maturity=True, maturity_scale=1e-12, recall=False)
    brain.basal_ganglia.maturity.use = Maturity.LIMIT
    stepped(brain, moments=5)
    assert brain.basal_ganglia.maturity.use <= Maturity.LIMIT
    with tempfile.TemporaryDirectory() as folder:
        loaded = cd.Brain.load(checkpoint(brain, folder))
    assert loaded.basal_ganglia.maturity == brain.basal_ganglia.maturity
