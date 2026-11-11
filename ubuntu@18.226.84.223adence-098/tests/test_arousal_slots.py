"""Extra exploration retains the other motor slots' acquired policy (#159)."""

from dataclasses import replace

import numpy as np
import pytest

import cadence as cd


def needy_brain(slots=5, seed=3):
    return cd.Brain.compose(
        2,
        3 * slots,
        modules=(8,),
        slots=slots,
        seed=seed,
        arousal=cd.ArousalConfig(youth=100, need=1.0), recall=False,
    )


def test_want_heats_one_slot_while_other_slots_keep_their_policy():
    brain = needy_brain()
    x = np.array([[0.2, 0.8]])
    heated = set()
    for moment in range(30):
        brain.live(x, **({"reward": [0.0]} if moment else {}))
        reading = brain.last_arousal
        temperatures = np.array(reading["temperatures"])
        slot = reading["heated_slot"]
        heated.add(slot)
        assert temperatures[slot] == reading["temperature"]
        other = np.arange(5) != slot
        assert (temperatures[other] == brain.learner.config.temperature).all()
        assert temperatures[slot] > brain.learner.config.temperature
        agent = brain.basal_ganglia
        actual = agent.probabilities(agent.state, temperatures)
        base = agent.probabilities(agent.state)
        np.testing.assert_array_equal(actual[:, other], base[:, other])
        assert (actual > 0).all()  # every motor command remains available
    assert heated == set(range(5))


def test_heating_one_slot_preserves_coordinated_choices_under_bounded_motor_margins():
    """An isolated motor-boundary control, not an acquisition or robot-skill claim."""
    base_weights = np.exp(np.array([1.0, 0.0, 0.0]) / 0.5)
    hot_weights = np.exp(np.array([1.0, 0.0, 0.0]) / 1.5)
    base, hot = base_weights / base_weights.sum(), hot_weights / hot_weights.sum()
    for slots in (3, 5, 8):
        brain = needy_brain(slots)
        brain.learner.config = replace(brain.learner.config, temperature=0.5)
        brain.live([[0.2, 0.8]])
        agent = brain.basal_ganglia
        free = agent.state
        activity = free.activation.copy()
        activity[:, brain.motor_index] = np.tile([1.0, 0.0, 0.0], slots)
        boundary = cd.BrainState(free.v, activity, free.adaptation, free.steps)
        actual = agent.probabilities(boundary, brain.last_arousal["temperatures"])
        all_heated = agent.probabilities(boundary, brain.last_arousal["temperature"])
        # Marginalize the uniformly selected heated slot, not a product of
        # averaged marginals: the extra exploration choices share one draw.
        whole_command = float(np.prod(actual[0, :, 0]))
        all_hot = float(np.prod(all_heated[0, :, 0]))
        assert whole_command == pytest.approx(hot[0] * base[0] ** (slots - 1))
        assert all_hot == pytest.approx(hot[0] ** slots)
        assert whole_command > all_hot
        assert base[0] ** slots > whole_command
        # Heating preserves coverage and increases alternatives in its slot.
        assert hot[1] > base[1] > 0


def test_refused_answer_draws_no_heated_slot_and_retry_matches_saved_twin(tmp_path):
    brain = needy_brain()
    twin = cd.Brain.load(brain.save(tmp_path / "before-action.npz"))
    config = brain.learner.config
    brain.learner.config = replace(config, free_steps=0)
    rng = brain.basal_ganglia.rng.bit_generator.state
    with pytest.raises(RuntimeError, match="did not settle"):
        brain.live([[0.2, 0.8]])
    assert brain.basal_ganglia.rng.bit_generator.state == rng
    assert brain.arousal.age == 0 and brain.last_arousal is None
    brain.learner.config = config
    np.testing.assert_array_equal(brain.live([[0.2, 0.8]]), twin.live([[0.2, 0.8]]))
    assert brain.last_arousal == twin.last_arousal


def test_accepted_feedback_then_refused_answer_keeps_no_extra_slot_draw(tmp_path, monkeypatch):
    brain = needy_brain()
    x = np.array([[0.2, 0.8]])
    brain.live(x)
    twin = cd.Brain.load(brain.save(tmp_path / "pending.npz"))
    settle = brain._settled
    rng = brain.basal_ganglia.rng.bit_generator.state

    def refuse(_):
        raise RuntimeError("the next answer refused")

    monkeypatch.setattr(brain, "_settled", refuse)
    with pytest.raises(RuntimeError, match="next answer refused"):
        brain.live(x, reward=[0.25])
    assert brain.basal_ganglia.updates == 1 and brain.arousal.rewards == 1
    assert brain.basal_ganglia.rng.bit_generator.state == rng
    monkeypatch.setattr(brain, "_settled", settle)
    np.testing.assert_array_equal(brain.live(x), twin.live(x, reward=[0.25]))
    assert brain.last_arousal["heated_slot"] == twin.last_arousal["heated_slot"]
    assert brain.last_arousal["temperatures"] == twin.last_arousal["temperatures"]
    # The accepted feedback's work was reported on the refused call; live's
    # successful-moment counters do not carry it into this reward-free retry.
    feedback_work = twin.arousal.learning_sweeps - brain.arousal.learning_sweeps
    assert feedback_work > 0

    def continuation(owner):
        state = owner.arousal.to_dict()
        if owner is brain:
            state["learning_sweeps"] += feedback_work
        return state

    assert continuation(brain) == continuation(twin)
    for moment in range(15):
        reward = [0.1 * (moment % 3)]
        np.testing.assert_array_equal(brain.live(x, reward=reward), twin.live(x, reward=reward))
        assert brain.last_arousal == twin.last_arousal
        assert continuation(brain) == continuation(twin)


@pytest.mark.parametrize("slots, heat", [(1, 0.0), (5, 0.0), (1, 2.0)])
def test_no_extra_heat_draw_without_extra_heat_or_with_only_one_slot(slots, heat):
    brain = needy_brain(slots)
    brain.arousal.config = replace(brain.arousal.config, heat=heat)
    x = np.array([[0.2, 0.8]])
    before = brain.basal_ganglia.rng.bit_generator.state
    brain.live(x)
    expected = np.random.default_rng()
    expected.bit_generator.state = before
    expected.random((1,) if slots == 1 else (1, slots))
    assert brain.basal_ganglia.rng.bit_generator.state == expected.bit_generator.state
    assert brain.last_arousal["heated_slot"] == (None if heat == 0 else 0)
    assert set(brain.last_arousal["temperatures"]) == {
        brain.learner.config.temperature * (1 + heat)
    }
