"""Income must remain observable while a multi-slot brain explores (#158)."""

import numpy as np
import pytest

import cadence as cd


def test_income_follows_actual_rewards_with_separate_own_error_calibration():
    arousal = cd.Arousal(cd.ArousalConfig(youth=0, fast=0.2, slow=0.1))
    rewards = np.array([0.8, -0.4, 0.3, 1.2, 0.2])
    own = [False, True, False, False, True]
    for index, (reward, chosen) in enumerate(zip(rewards, own, strict=True), 1):
        surprise, _ = arousal.outcome(0.25 if chosen else 20.0, reward, own=chosen)
        if not chosen:
            assert surprise == 0.0
        # Independent normalized exponential weights; sampled rewards cannot vanish.
        for reading, rate in ((arousal.recent, 0.2), (arousal.longrun, 0.1)):
            weights = (1 - rate) ** np.arange(index - 1, -1, -1)
            assert reading == pytest.approx(np.dot(weights, rewards[:index]) / weights.sum())
        assert arousal.rewards == index
        assert arousal.outcomes == sum(own[:index])
        assert arousal.usual == pytest.approx(0.25 if arousal.outcomes else 0.0)


def test_exploration_does_not_freeze_a_richer_stages_income_or_false_want():
    arousal = cd.Arousal(cd.ArousalConfig(youth=0))
    for moment in range(200):
        arousal.outcome(0.0, float(moment % 2))
    for _ in range(600):
        arousal.outcome(0.0, 0.5, learned=False)
    # The income falls before sampling begins. Previously these following sampled
    # outcomes froze the shortfall and kept the brain aroused indefinitely.
    for _ in range(60):
        arousal.outcome(0.0, 0.25)
    assert arousal.aroused and arousal.want > 0.2
    own_outcomes = arousal.outcomes
    modes = []
    for _ in range(1500):
        surprise, _ = arousal.outcome(9.0, 0.25, own=False)
        assert surprise == 0.0
        modes.append(arousal.aroused)
    assert arousal.outcomes == own_outcomes  # surprise retains its own-only contract
    assert arousal.rewards == own_outcomes + 1500
    assert arousal.recent == pytest.approx(0.25)
    assert arousal.longrun == pytest.approx(0.25, abs=0.0002)
    assert not any(modes[-300:]) and arousal.want < 0.05


def test_reward_origin_is_irrelevant_even_when_a_stream_starts_with_exploration():
    plain = cd.Arousal(cd.ArousalConfig(youth=0))
    shifted = cd.Arousal(plain.config)
    for index, reward in enumerate([0.2, 0.0, 0.4, -0.1] * 50):
        own = index % 7 == 6
        a = plain.outcome(0.1, reward, own=own)
        b = shifted.outcome(0.1, reward + 7, own=own)
        assert a == pytest.approx(b, abs=1e-11)
        assert plain.level == pytest.approx(shifted.level, abs=1e-11)
        assert plain.scale == pytest.approx(shifted.scale, abs=1e-11)


def test_a_genuine_unmet_need_remains_aroused_while_sampling():
    arousal = cd.Arousal(cd.ArousalConfig(youth=0, need=0.25))
    for _ in range(100):
        arousal.outcome(0.0, 0.5)
    for _ in range(2000):
        assert arousal.outcome(0.0, 0.0, own=False)[0] == 0.0
    assert arousal.aroused and arousal.want == 1.0
    assert arousal.rewards == 2100 and arousal.outcomes == 100


@pytest.mark.parametrize("count", [None, True, -1, 0.5, "2", 0])
def test_new_saved_income_refuses_a_missing_invalid_or_inconsistent_count(count):
    arousal = cd.Arousal()
    arousal.outcome(0.2, 0.5)
    saved = arousal.to_dict()
    if count is None:
        del saved["rewards"]
    else:
        saved["rewards"] = count
    with pytest.raises(ValueError, match="saved arousal"):
        cd.Arousal.from_dict(saved)


def test_slotted_sampling_reports_every_income_and_preserves_pending_continuation(tmp_path):
    brain = cd.Brain.compose(
        2,
        15,
        modules=(8,),
        slots=5,
        seed=3,
        arousal=cd.ArousalConfig(youth=100),
    )
    x = np.array([[0.2, 0.8]])
    brain.live(x)
    owned = 0
    for moment in range(40):
        owned += brain._lived[5]
        brain.live(x, reward=[0.1 * (moment % 2)])
        assert brain.arousal.rewards == moment + 1
        assert brain.arousal.outcomes == owned
    assert owned < 10  # whole-action agreement is rare, but income is not withheld
    assert brain.arousal.recent > 0 and brain.arousal.longrun > 0
    twin = cd.Brain.load(brain.save(tmp_path / "sampled-life.npz"))
    for moment in range(20):
        reward = [0.2 * (moment % 2)]
        np.testing.assert_array_equal(brain.live(x, reward=reward), twin.live(x, reward=reward))
        assert brain.last_arousal == twin.last_arousal
        assert brain.arousal.to_dict() == twin.arousal.to_dict()
    np.testing.assert_array_equal(brain.brain.efficacy, twin.brain.efficacy)
    np.testing.assert_array_equal(brain.hippocampus.consolidated, twin.hippocampus.consolidated)
