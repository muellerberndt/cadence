"""Adversarial numerical checks for accepted arousal genes and policy temperatures."""

import numpy as np
import pytest

import cadence as cd


def test_checkpoint_cannot_silently_replace_a_missing_arousal_gene_with_its_default():
    # a save carries every gene (cadence-arousal/4), at the founder need scale of zero as above
    # it, and misses none
    for config in (cd.ArousalConfig(youth=7, heat=3.0),
                   cd.ArousalConfig(youth=7, heat=3.0, need_scale=0.2)):
        arousal = cd.Arousal(config)
        for missing in arousal.to_dict()["config"]:
            saved = arousal.to_dict()
            del saved["config"][missing]
            with pytest.raises(ValueError, match="incomplete saved arousal config"):
                cd.Arousal.from_dict(saved)


def test_a_positive_rate_below_machine_epsilon_keeps_bias_corrected_readings_finite():
    arousal = cd.Arousal(cd.ArousalConfig(fast=1e-20, slow=1e-20, youth=0))
    arousal.outcome(0.25, 1.0)
    assert arousal.recent == pytest.approx(1.0)
    assert arousal.longrun == pytest.approx(1.0)
    assert arousal.usual == pytest.approx(0.25)
    assert arousal.scale == 0.0
    arousal.outcome(0.5, 0.0)
    assert arousal.recent == pytest.approx(0.5)
    assert arousal.longrun == pytest.approx(0.5)
    assert arousal.usual == pytest.approx(0.375)
    assert arousal.scale == pytest.approx(np.sqrt(0.125))
    assert cd.Arousal.from_dict(arousal.to_dict()).to_dict() == arousal.to_dict()


def test_surprise_can_compare_finite_errors_whose_ratio_overflows():
    arousal = cd.Arousal(cd.ArousalConfig(youth=0, tolerance=0.0, floor=0.0))
    arousal.outcome(0.0, 0.0)
    surprise, want = arousal.outcome(1e308, 0.0)
    assert surprise == pytest.approx(np.log(1e308) - np.log(1e-12))
    assert want == 0.0 and np.isfinite(arousal.level)
    assert cd.Arousal.from_dict(arousal.to_dict()).to_dict() == arousal.to_dict()


@pytest.mark.parametrize("own", [False, True])
def test_unrepresentable_reward_spread_refuses_without_changing_arousal(own):
    arousal = cd.Arousal(cd.ArousalConfig(youth=0))
    arousal.outcome(0.1, 0.5)
    arousal.lived(4, 2)
    before = arousal.to_dict()
    with pytest.raises(ValueError, match="moments must remain finite"):
        arousal.outcome(0.2, 1e200, own=own)
    assert arousal.to_dict() == before
    twin = cd.Arousal.from_dict(before)
    assert arousal.outcome(0.2, 0.25) == twin.outcome(0.2, 0.25)
    assert arousal.to_dict() == twin.to_dict()


@pytest.mark.parametrize("slots", [1, (2, 3)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_subnormal_temperature_keeps_categorical_slots_normalized(slots, dtype):
    brain = cd.Brain.compose(2, 5, modules=(8,), slots=slots, seed=0, recall=False)
    x = np.array([[0.2, 0.8]])
    expected = brain.act(x, greedy=True)
    agent = brain.basal_ganglia
    state = agent.state
    reading = cd.BrainState(state.v, state.activation.astype(dtype), state.adaptation, state.steps)
    p = agent.probabilities(reading, float(np.nextafter(0.0, 1.0)))
    assert np.isfinite(p).all()
    np.testing.assert_array_equal(p.sum(axis=-1), 1.0)
    np.testing.assert_array_equal(p.argmax(axis=-1), expected)
    action = brain.act(x, temperature=np.nextafter(0.0, 1.0))
    current = agent.probabilities(agent.state, np.nextafter(0.0, 1.0))
    np.testing.assert_array_equal(action, current.argmax(axis=-1))
    assert agent._pending is not None
