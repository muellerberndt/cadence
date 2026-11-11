"""Causal retention and continued adaptation, with no stored episode replay."""

import numpy as np
import pytest

import cadence as cd


def memory():
    return cd.SynapticMemory(np.arange(3), np.arange(3, 5))


def test_repetition_and_salience_consolidate_after_transient_memory_is_erased():
    once, repeated, salient = memory(), memory(), memory()
    cue, value = np.eye(3)[[0]], np.array([[1.0, 0.0]])
    once.observe(cue, value)
    for _ in range(40):
        repeated.observe(cue, value)
    salient.observe(cue, value, salience=np.array([19.0]))
    for m in (once, repeated, salient):
        # A fresh batch erases all fast residuals but retains shared slow synapses.
        m.reset(2)
    strengths = [m.recall(np.repeat(cue, 2, axis=0))[0, 0] for m in (once, repeated, salient)]
    np.testing.assert_allclose(strengths, [0.05, 1 - 0.95**40, 1])
    assert strengths[0] < 0.1 < 0.85 < strengths[1] < strengths[2]


def test_consolidation_survives_distraction_and_can_relearn_a_changed_outcome():
    m = memory()
    cue, other = np.eye(3)[[0]], np.eye(3)[[1]]
    m.observe(cue, np.array([[1.0, 0.0]]), salience=np.array([19.0]))
    for _ in range(200):
        m.observe(other, np.array([[0.0, 1.0]]))
    np.testing.assert_allclose(m.recall(cue), [[1, 0]])
    for _ in range(60):
        m.observe(cue, np.array([[0.0, 1.0]]))
    m.reset(1)
    assert m.recall(cue)[0, 1] > 0.95
    assert m.recall(other)[0, 1] > 0.99
    m.clear()
    np.testing.assert_array_equal(m.recall(cue), [[0, 0]])


def test_only_observed_values_consolidate_and_invalid_events_are_atomic():
    m = memory()
    cue = np.eye(3)[[0]]
    m.observe(cue, np.array([[1.0, 99.0]]), value_mask=np.array([[True, False]]))
    np.testing.assert_array_equal(m.consolidated[:, 1], 0)
    before = m.strength.copy(), m.consolidated.copy(), m.writes
    for options in ({"salience": [-1]}, {"value_mask": [[1, 0]]}, {"salience": [np.nan]}):
        with pytest.raises(ValueError):
            m.observe(cue, np.array([[0.0, 1.0]]), **options)
        np.testing.assert_array_equal(m.strength, before[0])
        np.testing.assert_array_equal(m.consolidated, before[1])
        assert m.writes == before[2]


def test_reads_do_not_rehearse_guesses_and_rows_keep_their_residuals():
    m = memory()
    keys = np.eye(3)[:2]
    m.observe(keys, np.eye(2))
    slow, fast = m.consolidated.copy(), m.strength.copy()
    for _ in range(100):
        m.recall(keys)
    np.testing.assert_array_equal(m.strength, fast)
    np.testing.assert_array_equal(m.consolidated, slow)
    m.reset(2, rows=np.array([0]))
    np.testing.assert_array_equal(m.strength[0], slow)
    np.testing.assert_array_equal(m.strength[1], fast[1])
    m.keep(np.array([1]))
    np.testing.assert_array_equal(m.strength[0], fast[1])
    # Resize + partial reset must initialize every new row from the slow weights.
    m.reset(3, rows=np.array([0]))
    np.testing.assert_array_equal(m.strength, np.broadcast_to(slow, (3, *slow.shape)))


def test_shared_slow_update_is_batch_order_invariant_and_correlated_keys_interfere():
    a, b = memory(), memory()
    keys = np.array([[1.0, 0, 0], [0.8, 0.6, 0]])
    values = np.eye(2)
    a.observe(keys, values)
    b.observe(keys[::-1], values[::-1])
    np.testing.assert_allclose(a.consolidated, b.consolidated)
    np.testing.assert_allclose(a.strength, b.strength[::-1])
    a.clear()
    a.observe(keys[:1], values[:1], salience=np.array([19.0]))
    a.observe(keys[1:], values[1:], salience=np.array([19.0]))
    a.reset(1)
    assert np.linalg.norm(a.recall(keys[:1]) - values[:1]) > 0.5


def test_no_writes_or_zero_keys_cannot_consolidate():
    m = memory()
    m.observe(np.zeros((1, 3)), np.ones((1, 2)), salience=np.array([1000.0]))
    m.observe(np.eye(3)[:1], np.ones((1, 2)), write=np.array([False]))
    assert m.writes == 0
    np.testing.assert_array_equal(m.consolidated, 0)


def test_demonstrations_can_enter_the_same_ongoing_loop():
    brain = cd.Brain.compose(2, 2, modules=(16,), seed=2, recall=False)
    for _ in range(400):
        brain.step(np.eye(2), teacher=np.arange(2))
    assert brain.accuracy(np.eye(2), np.arange(2)) == 1
    assert brain.learner.contrast_updates == 400
    assert brain.learner.updates == 799
    assert brain.basal_ganglia.updates == 399


@pytest.mark.parametrize("backend", ["cpu", "torch"])
def test_reward_updates_do_not_advance_imitation_optimizer_history(tmp_path, backend):
    if backend not in cd.available_backends():
        pytest.skip(backend)
    options = dict(
        modules=(4,), seed=0, backend=backend,
        learning=cd.LearnerConfig(momentum=0.9, normalize=0.95),
    )
    if backend == "torch":
        options["device"] = "cpu"
    a, b = (cd.Brain.compose(2, 2, **{"recall": False, **options}).learner for _ in range(2))
    drive = np.pad(np.eye(2), ((0, 0), (0, a.brain.connectome.n - 2)))
    for i in range(4):
        a.apply(np.zeros(a.brain.connectome.synapses), np.zeros(a.brain.connectome.n))
        if i == 2:
            a = cd.Learner.load(
                a.save(tmp_path / "optimizer.npz"), backend=backend, device=options.get("device")
            )
        a.step(drive, np.arange(2))
        b.step(drive, np.arange(2))
        np.testing.assert_allclose(a.brain.efficacy, b.brain.efficacy, atol=1e-12)
    assert a.contrast_updates == b.contrast_updates == 4
    assert a.updates == 8 and b.updates == 4


def test_unobserved_action_values_are_not_consolidated_by_the_reward_loop():
    brain = cd.Brain.compose(2, 3, modules=(4,), episodic=True, recall=False)
    x = np.eye(2)[:1]
    action = brain.step(x)[0]
    brain.step(x, reward=np.array([2.0]))
    unobserved = np.arange(3) != action
    np.testing.assert_array_equal(brain.hippocampus.consolidated[:, unobserved], 0)


@pytest.mark.parametrize("backend", ["cpu", "torch"])
def test_continuous_step_matches_explicit_loop_and_resumes_pending_action(tmp_path, backend):
    if backend not in cd.available_backends():
        pytest.skip(backend)
    options = dict(modules=(6,), episodic=True, seed=4, backend=backend)
    if backend == "torch":
        options["device"] = "cpu"
    ongoing = cd.Brain.compose(3, 2, **{"recall": False, **options})
    explicit = cd.Brain.compose(3, 2, **{"recall": False, **options})
    x = np.eye(3)[:2]
    np.testing.assert_array_equal(ongoing.step(x), explicit.act(x))
    for i in range(3):
        reward, done = np.array([1.0, -0.2]), np.array([i == 2, False])
        explicit.learn(reward, done, x)
        np.testing.assert_array_equal(ongoing.step(x, reward=reward, done=done), explicit.act(x))
    saved = ongoing.save(tmp_path / "live.npz")
    restored = cd.Brain.load(saved, backend=backend, device=options.get("device"))
    np.testing.assert_array_equal(
        ongoing.hippocampus.consolidated, restored.hippocampus.consolidated
    )
    for _ in range(3):
        for a in (ongoing, restored):
            a.step(x, reward=np.array([0.7, 0.0]))
        np.testing.assert_allclose(ongoing.brain.efficacy, restored.brain.efficacy, atol=1e-10)
        np.testing.assert_allclose(ongoing.hippocampus.strength, restored.hippocampus.strength)


def test_bad_current_demonstration_cannot_apply_previous_feedback():
    brain = cd.Brain.compose(2, 2, modules=(4,), episodic=True, recall=False)
    brain.step(np.eye(2))
    before = brain.brain.efficacy.copy()
    with pytest.raises(ValueError):
        brain.step(np.eye(2), reward=np.ones(2), teacher=[0, 0.5])
    assert brain.basal_ganglia.updates == 0
    assert brain.hippocampus.writes == 0
    np.testing.assert_array_equal(brain.brain.efficacy, before)
    brain.step(np.eye(2), reward=np.ones(2))
    assert brain.basal_ganglia.updates == 1


def test_unified_loop_learns_without_a_training_mode():
    brain = cd.Brain.compose(
        4, 4, seed=0, reward=cd.ActorCriticConfig(gamma=0, lam=0, eta=1, eta_critic=0.3), recall=False
    )
    rng = np.random.default_rng(0)
    cue = rng.integers(4, size=32)
    action = brain.step(np.eye(4)[cue])
    initial = brain.accuracy(np.eye(4), np.arange(4))
    for _ in range(400):
        reward = (action == cue).astype(float)
        cue = rng.integers(4, size=32)
        action = brain.step(np.eye(4)[cue], reward=reward, done=np.ones(32, bool))
    assert brain.accuracy(np.eye(4), np.arange(4)) >= 0.9 > initial
    assert brain.basal_ganglia.updates == 400
