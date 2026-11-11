"""What ``Brain.compose(..., recall=True)`` gives a new brain (``Recall.COMPOSED``, the lessons'
settings and a wide association's rates): each is an ordinary setting a caller's value overrides,
``recall_founder=True`` restores the founders, and a saved brain keeps the values it was saved with."""
from __future__ import annotations

import json

import numpy as np
import pytest

import cadence as cd
from cadence import generic

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
    grown_founder, grown_new = (b.grow(history_decays=(0.0, 0.5)) for b in (founder, new))
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


def test_every_new_brain_holds_the_memory_and_the_opt_out_builds_none(tmp_path):
    brain = cd.Brain.compose(6, 4, modules=(16,), seed=5)
    assert brain.recall is not None and brain.recall.anchor
    without = cd.Brain.compose(6, 4, modules=(16,), seed=5, recall=False)
    assert without.recall is None
    assert cd.Brain.load(without.save(tmp_path / "without.npz")).recall is None
    assert without.grow(history_decays=(0.0, 0.5)).recall is None
    assert without.grow(recall=True).recall is not None


def test_a_custom_anatomy_holds_a_memory_only_by_name():
    connectome = cd.develop(cd.Genome.composed(6, 4, modules=(16,)), seed=0)
    assert "recall" not in connectome.populations
    assert cd.Brain.compose(6, 4, connectome=connectome, seed=0).recall is None
    holding = cd.develop(cd.Genome.composed(6, 4, modules=(16,), recall=True), seed=0)
    assert cd.Brain.compose(6, 4, connectome=holding, seed=0, recall=True).recall is not None


def test_above_the_wide_limit_compose_names_the_three_ways_out(monkeypatch):
    monkeypatch.setattr(cd.Recall, "WIDE", 32)
    with pytest.raises(ValueError, match=r"code=units.*recall=False.*recall=True"):
        cd.Brain.compose(40, 2, modules=(8,), seed=0)
    assert cd.Brain.compose(32, 2, modules=(8,), seed=0).recall is not None
    assert cd.Brain.compose(40, 2, modules=(8,), seed=0, recall=False).recall is None
    assert cd.Brain.compose(40, 2, modules=(8,), seed=0, recall=True).recall is not None
    assert cd.Brain.compose(40, 2, modules=(8,), seed=0, code=32).recall is not None


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
    assert genes["learning_fast_settle"] is True
    assert genes["working_memory_amplitude"] == 3.0 and genes["recall_drive_gain"] is True
    assert genes["history_decays"] == [0.0]
    assert brain.carried_gains == {"trace": 1.0, "history_0": 1.0, "read": 1.0}
    assert composed(learning_surprise=False).describe()["genes"]["learning_surprise"] is False
    assert composed(recall_drive_gain=False).carried_gains == {"trace": 1.0, "history_0": 1.0}
    assert composed(history_decays=()).carried_gains == {"trace": 1.0, "read": 1.0}
    silent = composed(carried_gain=0.0)
    assert silent.hearing is None and not silent.recall.drive_gain
    assert composed(carried_gain=0.1).describe()["genes"]["carried_gain"] == 0.1
    founder = composed(recall_founder=True)
    assert founder.describe()["genes"]["learning_surprise"] is False and founder.hearing is None
    assert not founder.recall.drive_gain
    assert "learning_fast_settle" not in founder.describe()["genes"]
    assert founder.grow(history_decays=(0.0, 0.5)).hearing is None
    grown = composed().grow(history_decays=(0.0, 0.5))
    assert grown.carried_gains == {"trace": 1.0, "read": 1.0, "history_0": 1.0, "history_1": 1.0}
    without = cd.Brain.compose(6, 4, modules=(16,), seed=5, recall=False)
    assert without.hearing is None
    assert without.grow(recall=True).hearing is None and not without.grow(recall=True).recall.drive_gain


@pytest.mark.skipif(not cd.brain._FUSED, reason="CADENCE_FUSED=0 reads the NumPy loop alone")
def test_a_new_memory_takes_the_fast_solve_until_a_word_names_another():
    brain = composed()
    assert brain.learner.config.fast_settle is True
    lessons(brain, 2)
    assert "row_sweeps" in brain.last_settlement
    named = composed(learning_fast_settle=True)
    the_same(lessons(named, 2), brain)
    off = composed(learning_fast_settle=False)
    assert off.learner.config.fast_settle is False and "learning_fast_settle" not in off.describe()["genes"]
    lessons(off, 2)
    assert "row_sweeps" not in off.last_settlement
    assert composed(teaching="normalized").learner.config.fast_settle is True
    for word in ({"recall_founder": True}, {"learning": cd.LearnerConfig()}):
        assert composed(**word).learner.config.fast_settle is False
    assert cd.Brain.compose(6, 4, modules=(16,), seed=5, recall=False).learner.config.fast_settle is False
    attention = cd.Brain.compose(8, 4, modules=(16,), seed=0, learned_code=16, attention=True)
    assert attention.learner.config.fast_settle is False


def test_a_saved_brain_and_a_grown_brain_keep_the_solve_they_have(tmp_path):
    new, off = composed(), composed(learning_fast_settle=False)
    assert cd.Brain.load(new.save(tmp_path / "new.npz")).learner.config.fast_settle is True
    assert cd.Brain.load(off.save(tmp_path / "off.npz")).learner.config.fast_settle is False
    assert new.grow(history_decays=(0.0, 0.5)).learner.config.fast_settle is True
    assert off.grow(history_decays=(0.0, 0.5)).learner.config.fast_settle is False
    without = cd.Brain.compose(6, 4, modules=(16,), seed=5, recall=False)
    assert without.grow(recall=True).learner.config.fast_settle is False
    assert off.retune(learning_fast_settle=True).learner.config.fast_settle is True


def test_the_constructor_and_the_attention_recipe_keep_their_own_settings():
    connectome = cd.develop(cd.Genome.composed(6, 4, modules=(16,), recall=True), seed=0)
    direct = cd.Brain(connectome, seed=0, recall=True)
    attention = cd.Brain.compose(8, 4, modules=(16,), seed=0, learned_code=16, attention=True)
    for name in cd.Recall.COMPOSED:
        assert getattr(direct.recall, name) == founders()[name], name
        if name not in ("recency",):  # the attention recipe names its own recency
            assert getattr(attention.recall, name) == founders()[name], name


# ---------------------------------------------------------------------- the width's rates


def wide(width=128, **options):
    return cd.Brain.compose(6, 4, modules=(width,), seed=5, **options)


def earlier(monkeypatch, width=128, **options):
    """The brain composed before the width's rates: the same composition with the rates held off."""
    with monkeypatch.context() as patch:
        patch.setattr(generic, "_COMPOSED_WIDTH", 10**9)
        return cd.Brain.compose(6, 4, modules=(width,), seed=5, **options)


def plain(width=128, **options):
    """A new brain with the earlier rates named: the address map's founder, whole steps."""
    brain = wide(width, recall_address=0.01, **options)
    brain.learner.synapse_rate = None
    return brain


def lessons(brain, count=4):
    rng = np.random.default_rng(2)
    for _ in range(count):
        brain.step(rng.random((3, 6)), teacher=rng.integers(0, 4, size=3))
    return brain


def between(brain, pre, post):
    """The synapses from population ``pre`` to population ``post``."""
    connectome = brain.connectome
    return np.isin(connectome.pre, connectome.populations[pre]) & np.isin(
        connectome.post, connectome.populations[post]
    )


def the_same(first, second):
    np.testing.assert_array_equal(first.brain.efficacy, second.brain.efficacy)
    np.testing.assert_array_equal(first.brain.bias, second.brain.bias)
    for one, other in zip(first.recall.head_maps, second.recall.head_maps, strict=True):
        for a, b in zip(one, other, strict=True):
            np.testing.assert_array_equal(a, b)


def fan_in_rates(brain, narrow):
    """Each projection's step scaled by its fan-in in ``narrow`` over its fan-in in ``brain``,
    capped at 1, a reciprocal pair at the lower of its two: the recipe a composed width stands for."""
    c, n = brain.connectome, narrow.connectome
    rate = np.ones(c.synapses)
    names = [k for k in c.populations if k in n.populations and len(c.populations[k])]

    def labels(connectome):
        label = np.full(connectome.n, -1, dtype=np.int64)
        for i, k in enumerate(names):
            label[np.asarray(connectome.populations[k], dtype=np.int64)] = i
        return label

    lw, ln = labels(c), labels(n)
    for i in range(len(names)):
        for j, post in enumerate(names):
            into_w = (lw[c.pre] == i) & (lw[c.post] == j)
            into_n = (ln[n.pre] == i) & (ln[n.post] == j)
            if into_w.any() and into_n.any():
                fan_w = into_w.sum() / len(c.populations[post])
                fan_n = into_n.sum() / len(n.populations[post])
                rate[into_w] = min(1.0, fan_n / fan_w)
    reverse = np.asarray(brain.learner.reverse)
    paired = reverse >= 0
    rate[paired] = np.minimum(rate[paired], rate[reverse[paired]])
    return rate


@pytest.mark.parametrize("width", [65, 96, 128, 256, 512])
def test_a_wide_memory_steps_each_growing_projection_at_its_width(width):
    brain = wide(width)
    np.testing.assert_array_equal(brain.learner.synapse_rate, fan_in_rates(brain, wide(64)))
    assert brain.recall.address == 0.01 * 64 / width
    grows = (
        between(brain, "prefrontal", "association")
        | between(brain, "association", "motor")
        | between(brain, "motor", "association")
    )
    assert (brain.learner.synapse_rate[grows] == 64 / width).all()
    assert (brain.learner.synapse_rate[~grows] == 1.0).all()


@pytest.mark.parametrize("width", [16, 63, 64])
def test_at_64_and_under_a_new_memory_holds_no_rates(width):
    brain = wide(width)
    assert brain.learner.synapse_rate is None
    assert brain.recall.address == 0.01
    assert "projection_rates" not in brain.describe()["layout"]


def test_the_association_sets_the_rates_and_no_other_projection_takes_one():
    chained = cd.Brain.compose(6, 4, modules=(32, 128), observers=(8,), seed=5)
    assert chained.recall.address == 0.01 * 64 / 128
    assert set(chained.describe()["layout"]["projection_rates"]) == {
        "prefrontal->association", "association->motor", "motor->association",
    }
    narrow_last = cd.Brain.compose(6, 4, modules=(128, 32), seed=5)
    assert narrow_last.learner.synapse_rate is None and narrow_last.recall.address == 0.01


def test_a_named_address_rate_wins_and_leaves_the_projections():
    brain = wide(recall_address=0.02)
    assert brain.recall.address == 0.02
    assert (brain.learner.synapse_rate[between(brain, "association", "motor")] == 0.5).all()
    assert wide(recall_address=0.0).recall.address == 0.0


@pytest.mark.parametrize(
    "named", [{"teaching": "founder"}, {"teaching": "normalized"}, {"learning": cd.LearnerConfig()}]
)
def test_named_lessons_win_on_the_projections_and_leave_the_read_map(named):
    brain = wide(**named)
    assert brain.learner.synapse_rate is None
    assert brain.recall.address == 0.01 * 64 / 128


def test_naming_the_deep_point_is_the_new_brain():
    named, unnamed = wide(teaching="deep"), wide()
    np.testing.assert_array_equal(named.learner.synapse_rate, unnamed.learner.synapse_rate)
    the_same(lessons(named), lessons(unnamed))


@pytest.mark.parametrize(
    "options",
    [
        {"recall": True, "recall_founder": True},
        {"recall": False},
    ],
)
def test_the_opt_outs_keep_whole_steps(options):
    brain = wide(**options)
    assert brain.learner.synapse_rate is None
    assert brain.recall is None or brain.recall.address == 0.01


def test_the_attention_recipe_and_a_custom_anatomy_take_no_rates():
    attention = cd.Brain.compose(8, 4, modules=(128,), seed=0, learned_code=16, attention=True)
    assert attention.learner.synapse_rate is None and attention.recall.address == 0.0
    holding = cd.develop(cd.Genome.composed(6, 4, modules=(128,), recall=True), seed=0)
    custom = cd.Brain.compose(6, 4, connectome=holding, seed=0, recall=True)
    assert custom.learner.synapse_rate is None and custom.recall.address == 0.01


def test_a_step_is_the_whole_step_times_its_rate():
    rated, whole = wide(recall_address=0.01), wide(recall_address=0.01)
    whole.learner.synapse_rate = None
    start = rated.brain.efficacy.copy()
    x = np.random.default_rng(3).random((8, 6))
    labels = np.arange(8) % 4
    rated.lesson(x, labels), whole.lesson(x, labels)
    moved = whole.brain.efficacy - start
    grows = (
        between(rated, "prefrontal", "association")
        | between(rated, "association", "motor")
        | between(rated, "motor", "association")
    )
    assert np.abs(moved[grows]).max() > 0
    np.testing.assert_allclose((rated.brain.efficacy - start)[grows], 0.5 * moved[grows], atol=1e-14)
    np.testing.assert_array_equal((rated.brain.efficacy - start)[~grows], moved[~grows])


def test_the_earlier_rates_named_are_the_earlier_brain(monkeypatch):
    the_same(lessons(plain()), lessons(earlier(monkeypatch)))
    assert plain().parameters() == wide().parameters()


def test_the_rates_are_described():
    brain = wide(256)
    described = brain.describe()
    assert described["genes"]["recall_address"] == 0.0025
    assert described["layout"]["projection_rates"] == {
        "association->motor": 0.25, "prefrontal->association": 0.25, "motor->association": 0.25,
    }
    json.dumps(described)
    assert "projection_rates" not in plain(256).describe()["layout"]
    by_hand = wide()
    by_hand.learner.synapse_rate = np.where(
        between(by_hand, "sensory", "association"), np.linspace(0.2, 0.4, by_hand.connectome.synapses), 1.0
    )
    rates = by_hand.describe()["layout"]["projection_rates"]
    assert list(rates) == ["sensory->association"]
    assert rates["sensory->association"][0] < rates["sensory->association"][1]


def test_a_checkpoint_keeps_the_rates_and_continues(tmp_path):
    brain = lessons(wide())
    twin = cd.Brain.load(brain.save(tmp_path / "wide.npz"))
    np.testing.assert_array_equal(twin.learner.synapse_rate, brain.learner.synapse_rate)
    assert twin.recall.address == brain.recall.address
    x = np.random.default_rng(4).random((3, 6))
    brain.reset(), twin.reset()
    for _ in range(3):
        brain.step(x, teacher=np.array([0, 1, 2]))
        twin.step(x, teacher=np.array([0, 1, 2]))
    the_same(brain, twin)


def test_a_brain_saved_under_the_earlier_rates_keeps_them(tmp_path):
    brain = lessons(plain())
    saved = cd.Brain.load(brain.save(tmp_path / "earlier.npz"))
    assert saved.learner.synapse_rate is None and saved.recall.address == 0.01
    assert "projection_rates" not in saved.describe()["layout"]
    grown = saved.grow(history_decays=(0.0, 0.5))
    assert grown.learner.synapse_rate is None and grown.recall.address == 0.01


def test_retune_keeps_the_rates_and_takes_a_named_address():
    brain = wide()
    rates = brain.learner.synapse_rate.copy()
    brain.retune(teaching="normalized", learning_scale_cap=4.0, recall_address=0.03)
    np.testing.assert_array_equal(brain.learner.synapse_rate, rates)
    assert brain.recall.address == 0.03
    with pytest.raises(TypeError, match="synapse_rate"):
        brain.retune(synapse_rate=rates)


@pytest.mark.parametrize(
    "parts",
    [
        {"history_decays": (0.0, 0.5)},
        {"efference_amplitude": 1.0},
        {"recall_heads": 2},
        {"history_decays": (0.0, 0.5), "recall_heads": 2},
    ],
)
def test_a_grown_brain_keeps_its_rates_and_a_new_part_steps_whole(parts):
    trained = lessons(wide())
    grown = trained.grow(**parts)
    assert grown.recall.address == trained.recall.address
    same = grown.describe()["layout"]["projection_rates"]
    assert same == trained.describe()["layout"]["projection_rates"]
    assert len(grown.learner.synapse_rate) == grown.connectome.synapses
    assert ((grown.learner.synapse_rate == 0.5).sum() == (trained.learner.synapse_rate == 0.5).sum())
    x = np.random.default_rng(5).random((3, 6))
    trained.reset()
    np.testing.assert_array_equal(trained.act(x, greedy=True), grown.act(x, greedy=True))


def test_a_new_slot_takes_the_pairs_step():
    trained = lessons(wide())
    grown = trained.grow(slots=(4, 2))
    np.testing.assert_array_equal(
        grown.learner.synapse_rate, generic._width_rates(grown.connectome)
    )
    assert (grown.learner.synapse_rate[between(grown, "association", "motor")] == 0.5).all()
    assert len(grown.connectome.populations["motor"]) == 6


def test_the_width_is_fixed_at_birth_so_growth_keeps_its_rates():
    trained = lessons(wide())
    with pytest.raises(ValueError, match="own loop"):
        trained.grow(modules=(256,))
    twice = trained.grow(slots=(4, 2)).grow(history_decays=(0.0, 0.5))
    np.testing.assert_array_equal(twice.learner.synapse_rate, generic._width_rates(twice.connectome))
    assert twice.recall.address == trained.recall.address
    assert lessons(plain()).grow(slots=(4, 2)).learner.synapse_rate is None


@pytest.mark.parametrize(
    "options",
    [{"observers": (8,)}, {"modules": (32, 96)}, {"recall_heads": 2}, {"recall_gain": 1.0, "recall_tokens": [0, 1, 2, 3]},
     {"recall_when": True}, {"history_decays": (0.0, 0.5)}, {"efference_amplitude": 1.0}, {"learned_code": 16}, {"apical": 0.25},
     {"arousal": True, "maturity": True}, {"memory_choice": True}, {"learning_rehearsal": 1.0}, {"inputs": (6, 6)},
     {"inputs": 64, "code": 32}],
    ids=lambda options: "+".join(sorted(options)),
)
def test_a_wide_new_brain_holds_its_rates_beside_every_part(options, tmp_path):
    options = dict(options)
    inputs = options.pop("inputs", 6)
    brain = cd.Brain.compose(inputs, 4, **{"modules": (96,), **options}, seed=5)
    rng = np.random.default_rng(6)
    for _ in range(3):
        brain.step(rng.random((3, int(np.prod(inputs)))), teacher=rng.integers(0, 4, size=3))
    assert brain.recall.address == 0.01 * 64 / 96
    assert set(brain.describe()["layout"]["projection_rates"].values()) == {64 / 96}
    again = cd.Brain.load(brain.save(tmp_path / "wide.npz"))
    np.testing.assert_array_equal(again.learner.synapse_rate, brain.learner.synapse_rate)


def test_a_grown_brain_saves_and_continues_at_its_rates(tmp_path):
    grown = lessons(wide()).grow(history_decays=(0.0, 0.5))
    twin = cd.Brain.load(grown.save(tmp_path / "grown.npz"))
    np.testing.assert_array_equal(twin.learner.synapse_rate, grown.learner.synapse_rate)
    lessons(grown), lessons(twin)
    the_same(grown, twin)


def test_growing_a_memory_keeps_the_brains_steps_and_takes_the_maps():
    without = lessons(wide(recall=False))
    assert without.learner.synapse_rate is None
    grown = without.grow(recall=True)
    assert grown.learner.synapse_rate is None
    assert grown.recall.address == 0.01 * 64 / 128
    assert without.grow(recall=True, recall_address=0.02).recall.address == 0.02
    assert without.grow(recall=True, recall_founder=True).recall.address == 0.01


def test_grow_and_graft_refuse_rates_set_by_hand():
    brain = wide()
    brain.learner.synapse_rate = np.full(brain.connectome.synapses, 0.5)
    with pytest.raises(ValueError, match="rates set by hand"):
        brain.grow(history_decays=(0.0, 0.5))
    with pytest.raises(ValueError, match="rates set by hand"):
        wide().graft(brain, regions=("association", "motor"))
    with pytest.raises(ValueError, match="rates set by hand"):
        brain.graft(wide(), regions=("association", "motor"))
    narrow = wide(64)
    narrow.learner.synapse_rate = np.ones(narrow.connectome.synapses)
    with pytest.raises(ValueError, match="rates set by hand"):
        narrow.grow(history_decays=(0.0, 0.5))


@pytest.mark.parametrize(
    ("host", "donor", "rated"),
    [(wide, wide, True), (wide, plain, True), (plain, wide, False), (plain, plain, False)],
)
def test_a_graft_steps_at_the_hosts_rates(host, donor, rated):
    target, source = host(), lessons(donor())
    grafted = target.graft(source, regions=("association", "motor"))
    if rated:
        np.testing.assert_array_equal(grafted.learner.synapse_rate, target.learner.synapse_rate)
    else:
        assert grafted.learner.synapse_rate is None
    assert grafted.recall.address == target.recall.address
    pair = between(grafted, "association", "motor")
    np.testing.assert_array_equal(grafted.brain.efficacy[pair], source.brain.efficacy[pair])
