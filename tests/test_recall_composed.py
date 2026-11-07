"""What ``Brain.compose(..., recall=True)`` gives a new brain (``Recall.COMPOSED`` and the lessons'
settings): each is an ordinary setting a caller's value overrides, ``recall_founder=True`` restores the
founders, and a saved brain keeps the values it was saved with."""
from __future__ import annotations

import numpy as np
import pytest

import cadence as cd

X = [np.random.default_rng(0).random((2, 6)) for _ in range(6)]


def composed(**options):
    return cd.Brain.compose(6, 4, modules=(16,), seed=5, recall=True, **options)


def founders() -> dict[str, object]:
    return {**cd.Recall.founders(), **cd.Recall.options()}


def test_the_founder_word_is_no_setting_of_the_memory():
    brain = composed(recall_founder=True)
    assert "recall_founder" not in brain.describe()["genes"]
    assert not hasattr(brain.recall, "founder")
    with pytest.raises(ValueError, match="recall_founder must be boolean"):
        composed(recall_founder=1)
    with pytest.raises(ValueError, match="recall settings need recall=True"):
        cd.Brain.compose(6, 4, seed=5, recall=False, recall_founder=True)


def test_the_founder_word_gives_every_setting_its_founder():
    brain = composed(recall_founder=True)
    for name, value in founders().items():
        assert getattr(brain.recall, name) == value, name


@pytest.mark.parametrize("name", sorted(cd.Recall.COMPOSED))
def test_a_new_memory_takes_what_is_composed_and_a_callers_value_overrides(name):
    assert getattr(composed().recall, name) == cd.Recall.COMPOSED[name]
    assert cd.Recall.COMPOSED[name] != founders()[name]
    kept = composed(**{f"recall_{name}": founders()[name]})
    assert getattr(kept.recall, name) == founders()[name]


def test_a_saved_memory_keeps_its_values(tmp_path):
    founder = composed(recall_founder=True)
    loaded = cd.Brain.load(founder.save(tmp_path / "founder.npz"))
    new = composed()
    again = cd.Brain.load(new.save(tmp_path / "new.npz"))
    for name in cd.Recall.COMPOSED:
        assert getattr(loaded.recall, name) == getattr(founder.recall, name), name
        assert getattr(again.recall, name) == cd.Recall.COMPOSED[name], name


def test_a_grown_brain_keeps_its_own_values_whatever_a_new_memory_composes():
    founder = composed(recall_founder=True)
    new = composed()
    for brain in (founder, new):
        rng = np.random.default_rng(2)
        for x in X[:3]:
            brain.lesson(x, rng.integers(0, 4, size=2))
    grown_founder, grown_new = (b.grow(history_decays=(0.5,)) for b in (founder, new))
    for name in cd.Recall.COMPOSED:
        assert getattr(grown_founder.recall, name) == getattr(founder.recall, name), name
        assert getattr(grown_new.recall, name) == cd.Recall.COMPOSED[name], name
    assert grown_founder.learner.config == founder.learner.config
    assert grown_new.learner.config == new.learner.config


def test_a_new_memory_teaches_at_the_deep_point_with_no_apical():
    genes = composed().describe()["genes"]
    assert genes["teaching"] == "deep" and composed().brain.apical is None
    assert composed(recall_founder=True).describe()["genes"]["teaching"] == "founder"
    assert composed(teaching="founder").describe()["genes"]["teaching"] == "founder"
    assert composed(teaching="normalized").describe()["genes"]["teaching"] == "normalized"
    assert composed(learning=cd.LearnerConfig()).describe()["genes"]["teaching"] == "custom"


@pytest.mark.parametrize(
    "options",
    [{"recall_position": 0}, {"recall_position_rate": 0.0}, {"recall_heads": 2},
     {"recall_window": 2}, {"recall_next": False},
     {"recall_action": True}, {"recall_when": True}, {"recall_capacity": 8},
     {"recall_copy": 2.0, "recall_tokens": [0, 1, 2, 3]},
     {"recall_gain": 1.0, "recall_tokens": [0, 1, 2, 3]},
     {"recall_address": 0.0}, {"recall_address_reward": 1.0}, {"recall_position_reward": 1.0},
     {"recall_grid": ()}, {"recall_position": 64}, {"carried_gain": 0.0}, {"recall_drive_gain": False}],
    ids=lambda options: "+".join(sorted(options)),
)
def test_every_setting_composes_beside_what_a_new_memory_takes_and_learns(options):
    brain = composed(**options)
    rng = np.random.default_rng(4)
    for x in X[:3]:
        brain.step(x, teacher=rng.integers(0, 4, size=2))
    brain.act(X[3])
    brain.learn(np.array([1.0, 0.0]), np.array([False, False]), X[4])


def test_no_position_term_leaves_out_what_reads_beside_it():
    brain = composed(recall_position=0)
    assert brain.recall.position == 0 and not brain.recall.position_maps
    assert not brain.recall.anchor and not brain.recall.relative and not brain.recall.slope
    assert not brain.recall.position_reward and not brain.recall.grid
    assert composed(recall_position_rate=0.0).recall.position_reward == 0.0


def test_a_new_memory_takes_the_settings_beside_it():
    brain = composed()
    genes = brain.describe()["genes"]
    assert genes["learning_surprise"] is True and genes["carried_gain"] == 0.01
    assert genes["working_memory_amplitude"] == 3.0 and genes["recall_drive_gain"] is True
    assert brain.carried_gains == {"trace": 1.0, "read": 1.0}
    assert composed(learning_surprise=False).describe()["genes"]["learning_surprise"] is False
    assert composed(recall_drive_gain=False).carried_gains == {"trace": 1.0}
    silent = composed(carried_gain=0.0)
    assert silent.hearing is None and not silent.recall.drive_gain
    assert composed(carried_gain=0.1).describe()["genes"]["carried_gain"] == 0.1
    founder = composed(recall_founder=True)
    assert founder.describe()["genes"]["learning_surprise"] is False and founder.hearing is None
    assert not founder.recall.drive_gain
    assert founder.grow(history_decays=(0.5,)).hearing is None
    grown = composed().grow(history_decays=(0.5,))
    assert grown.carried_gains == {"trace": 1.0, "read": 1.0, "history_0": 1.0}
    without = cd.Brain.compose(6, 4, modules=(16,), seed=5, recall=False)
    assert without.hearing is None
    assert without.grow(recall=True).hearing is None and not without.grow(recall=True).recall.drive_gain


def test_the_constructor_and_the_attention_recipe_keep_their_own_settings():
    connectome = cd.develop(cd.Genome.composed(6, 4, modules=(16,), recall=True), seed=0)
    direct = cd.Brain(connectome, seed=0, recall=True)
    attention = cd.Brain.compose(8, 4, modules=(16,), seed=0, learned_code=16, attention=True)
    for name in cd.Recall.COMPOSED:
        assert getattr(direct.recall, name) == founders()[name], name
        if name not in ("recency",):  # the attention recipe names its own recency
            assert getattr(attention.recall, name) == founders()[name], name
