"""The generic brain learns simple tasks from labels, pictures, reward and one trial."""

from __future__ import annotations

import numpy as np
import pytest

import cadence as cd


def blobs(n_per: int = 60, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.zeros((2 * n_per, 8))
    x[:n_per, :4] = 1.0
    x[n_per:, 4:] = 1.0
    return np.clip(x + 0.3 * rng.standard_normal(x.shape), 0, 1), np.repeat([0, 1], n_per)


def bars(n_per: int = 40, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """6 by 6 pictures of one vertical or one horizontal bar at a random place, with noise."""
    rng = np.random.default_rng(seed)
    pictures = np.zeros((2 * n_per, 6, 6))
    for k in range(2 * n_per):
        at = rng.integers(1, 5)
        if k < n_per:
            pictures[k, :, at] = 1.0
        else:
            pictures[k, at, :] = 1.0
    pictures = np.clip(pictures + 0.2 * rng.standard_normal(pictures.shape), 0, 1)
    return pictures, np.repeat([0, 1], n_per)


def test_the_generic_brain_learns_labels() -> None:
    # defaults (the single-region layout of earlier releases reached 1.0 on held-out
    # samples on ten seeds of this task)
    brain = cd.Brain.compose(8, 2, seed=1)
    x, y = blobs()
    history = brain.fit(x, y, epochs=4, batch=20)
    held_x, held_y = blobs(seed=7)
    assert history[-1] >= 0.95 and brain.accuracy(held_x, held_y) >= 0.95


def test_the_generic_brain_sees_bars_through_its_visual_cortex() -> None:
    # defaults (on the single-region layout of earlier releases the held-out accuracy over
    # ten seeds had median 0.97 and minimum 0.89)
    brain = cd.Brain.compose((6, 6), 2, seed=0)
    assert "visual/input" in brain.connectome.populations
    x, y = bars()
    brain.fit(x, y, epochs=30, batch=20)
    held_x, held_y = bars(seed=100)
    assert brain.accuracy(held_x, held_y) >= 0.85


def test_the_generic_brain_learns_a_contextual_bandit_from_dopamine() -> None:
    # defaults with an immediate reward (the single-region layout of earlier releases reached
    # 1.0 on ten seeds of this task)
    reward_config = cd.ActorCriticConfig(gamma=0.0, lam=0.0, eta=1.0, eta_critic=0.3)
    brain = cd.Brain.compose(4, 4, seed=0, reward=reward_config)
    rng = np.random.default_rng(0)

    def contexts(k: int) -> tuple[np.ndarray, np.ndarray]:
        c = rng.integers(0, 4, size=k)
        return np.eye(4)[c], c

    def hit_rate() -> float:
        x, c = contexts(200)
        brain.reset()
        return float((brain.act(x, greedy=True) == c).mean())

    before = hit_rate()
    brain.reset()
    x, c = contexts(32)
    for _ in range(400):
        action = brain.act(x)
        reward = (action == c).astype(float)
        x, c = contexts(32)
        brain.learn(reward, np.ones(32, dtype=bool), x)
    assert hit_rate() >= 0.9 > before


def test_the_hippocampus_keeps_a_rewarded_choice_after_one_trial() -> None:
    """Assigned-record/readout control; actual sampled-action custody is tested separately."""
    frozen = cd.LearnerConfig(eta=0.0, eta_bias=0.0, tolerance=3e-3, nudged_steps=12)
    still = cd.ActorCriticConfig(gamma=0.0, lam=0.0, eta=0.0, eta_bias=0.0, eta_critic=0.0)
    brain = cd.Brain.compose(
        4, 4, modules=(8,), episodic=True, seed=3, learning=frozen, reward=still
    )
    cue, other = np.eye(4)[[2]], np.eye(4)[[0]]
    first = int(brain.act(cue, greedy=True)[0])
    rewarded = (first + 1) % 4
    brain.reset()
    brain.act(cue)  # explore
    # Assign a different record to isolate its influence on the otherwise frozen readout.
    # This is a component control, not an executed action or a real outcome history.
    brain._moment = (cue, np.array([rewarded]))
    brain.learn(np.array([1.0]), np.array([True]), cue)
    brain.reset()
    assert int(brain.act(cue, greedy=True)[0]) == rewarded
    brain.reset()
    assert int(brain.act(other, greedy=True)[0]) == int(
        cd.Brain.compose(4, 4, modules=(8,), seed=3, learning=frozen, reward=still).act(
            other, greedy=True
        )[0]
    )


def delayed_response(brain: cd.Brain, episodes: int, seed: int) -> float:
    """A cue, then a go signal with the cue gone; reward for the action that names the cue."""
    rng = np.random.default_rng(seed)
    batch = 32

    def episode() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cue = rng.integers(0, 2, size=batch)
        shown, go = np.zeros((batch, 3)), np.zeros((batch, 3))
        shown[np.arange(batch), cue] = 1.0
        go[:, 2] = 1.0
        return cue, shown, go

    for _ in range(episodes):
        cue, shown, go = episode()
        brain.reset()
        brain.act(shown)
        brain.learn(np.zeros(batch), np.zeros(batch, dtype=bool), go)
        action = brain.act(go)
        brain.learn((action == cue).astype(float), np.ones(batch, dtype=bool), go)
    hits = []
    for _ in range(5):
        cue, shown, go = episode()
        brain.reset()
        brain.act(shown, greedy=True)
        hits.append(float((brain.act(go, greedy=True) == cue).mean()))
    return float(np.mean(hits))


def test_the_prefrontal_cortex_bridges_a_delay() -> None:
    # defaults (tests/test_context_defaults.py reads seeds 8 to 13 at 0.96 to 1.00); the same
    # brain with its working trace silenced is the control
    remembering = cd.Brain.compose(3, 2, seed=8)
    assert delayed_response(remembering, episodes=1500, seed=8) >= 0.9
    forgetting = cd.Brain.compose(3, 2, seed=8, working_memory_amplitude=0.0)
    assert delayed_response(forgetting, episodes=400, seed=8) < 0.75


@pytest.mark.parametrize("slots", [2, [3, 1]])
def test_slotted_brain_predict_accuracy_and_fit_match_the_slot_contract(
    slots: int | list[int],
) -> None:
    """Slotted readouts answer one choice per slot everywhere, or the slot API is a lie."""
    actions = sum(slots) if isinstance(slots, list) else 2 * slots
    brain = cd.Brain.compose(4, actions, modules=(16,), slots=slots, seed=3)
    rng = np.random.default_rng(3)
    x = rng.normal(size=(30, 4))
    y = np.stack(
        [rng.integers(0, size, size=30) for size in brain.learner.slot_sizes], axis=1
    )
    brain.fit(x[:10], y[:10], epochs=1, batch=5)

    predicted = brain.predict(x[10:20])
    assert predicted.shape == (10, brain.learner.slot_count)
    drive = brain.stimulus(x[10:20], memory=False)
    per_slot = brain.learner.predict(drive)
    # predict settles under the brain's admission, the learner under its own free phase;
    # both qualify at the tolerance, so their choices agree wherever a slot's two most
    # active motor neurons lie further apart than the two residuals together (without
    # motor competition two of them can tie within it); a slot of one neuron has one choice
    out = np.asarray(brain.learner.free(drive).activation)[:, brain.motor_index]
    tolerance = brain.learner.config.tolerance
    assert tolerance is not None
    for slot, (offset, size) in enumerate(
        zip(brain.learner.slot_offsets, brain.learner.slot_sizes, strict=True)
    ):
        top = np.sort(out[:, offset : offset + size], axis=1)
        clear = np.ones(len(top), dtype=bool)
        if size > 1:
            clear = top[:, -1] - top[:, -2] > 2 * tolerance
        assert clear.any()
        np.testing.assert_array_equal(predicted[clear, slot], per_slot[clear, slot])
    # act reads memory, so its choices may differ; its shape and slot bounds may not
    actions = brain.act(x[10:20], greedy=True)
    assert actions.shape == (10, brain.learner.slot_count)
    assert (
        actions < np.asarray(brain.learner.slot_sizes)[None, :]
    ).all() and (actions >= 0).all()

    # accuracy counts the choices of predict, one per slot
    assert brain.accuracy(x[10:20], y[10:20]) == float(np.mean(predicted == y[10:20]))

    history = brain.fit(x[20:], y[20:], epochs=1, batch=5)
    assert len(history) == 1 and 0.0 <= history[0] <= 1.0


def test_fit_mismatched_labels_fail_before_teaching() -> None:
    brain = cd.Brain.compose(4, 2, modules=(8,), seed=1)
    x = np.zeros((4, 4))
    before = brain.brain.efficacy.copy()
    with pytest.raises(ValueError, match="same batch size"):
        brain.fit(x, np.array([0], dtype=int))
    np.testing.assert_array_equal(brain.brain.efficacy, before)
    assert brain.learner.updates == 0


def test_evolution_selects_a_generic_brain_genome() -> None:
    x, y = blobs(30)

    def fitness(connectome: cd.Connectome, seed: int) -> float:
        brain = cd.Brain.compose(8, 2, connectome=connectome, seed=seed)
        brain.fit(x, y, epochs=1, batch=20)
        return brain.accuracy(x, y) - 1e-3 * len(connectome.populations["association"])

    genome = cd.Genome.composed(8, 2, modules=(6,))
    lineage = cd.evolve(
        fitness, genome, generations=2, population=3, keep=1, seed=0, fixed=("sensory",),
        tied=(("association", "prefrontal"),),
    )
    assert lineage.best is not None and lineage.best.region("motor").size == 2


def test_a_lesson_on_the_moment_just_acted_on_warns_with_a_recall_memory():
    """An act writes its moment to the recall memory, so a lesson on the same observations
    reads the moment itself, its successor still empty, and warns; the next moment taught
    before its act, a lesson after a reset, and a brain without a recall memory do not."""
    import warnings

    obs = np.array([[1.0, 0.0, 0.0]])
    brain = cd.Brain.compose(3, 2, modules=(8,), seed=0, learned_code=4, attention=True)
    brain.act(obs, greedy=True)
    with pytest.warns(RuntimeWarning, match="Teach before acting"):
        brain.lesson(obs, [1])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        other = np.array([[0.0, 1.0, 0.0]])
        brain.lesson(other, [0])  # the next moment, taught before its act
        brain.act(other, greedy=True)
        brain.reset()
        brain.lesson(obs, [1])
        plain = cd.Brain.compose(3, 2, modules=(8,), seed=0)  # no recall memory: no warning
        plain.act(obs, greedy=True)
        plain.lesson(obs, [1])
