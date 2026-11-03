"""Public configuration preserves the acquired life and the ownership of its outcomes."""

import json
from dataclasses import replace

import numpy as np
import pytest
from checkpoints import earlier, metadata

import cadence as cd

OBSERVATIONS = np.eye(3)


def small(**options):
    settings = dict(
        modules=(8,),
        seed=3,
        arousal=True,
        working_memory_amplitude=0.3,
        actor_eta=0.03,
        actor_eta_bias=0.003,
    )
    settings.update(options)
    return cd.Brain.compose(3, 2, **settings)


def archive(brain, tmp_path, name):
    with np.load(brain.save(tmp_path / name), allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def assert_archives_equal(left, right):
    assert left.keys() == right.keys()
    for name in left:
        np.testing.assert_array_equal(left[name], right[name], err_msg=name)


def acquire(brain):
    action = brain.live(OBSERVATIONS[[0]])
    for index in range(1, 9):
        reward = float(action[0] == (index - 1) % 2)
        action = brain.live(OBSERVATIONS[[index % 3]], reward=[reward])
    return action


def test_plain_compose_keeps_the_released_founders():
    brain = cd.Brain.compose(3, 2, modules=(8,))
    assert brain.arousal is None and brain.efference is None
    assert brain.learner.config.eta == 0.5
    assert brain.learner.config.eta_bias == 0.05
    assert brain.basal_ganglia.config.eta == 1.0
    assert brain.basal_ganglia.config.eta_bias == 0.05
    assert brain.working_memory.amplitude == 3.0
    assert brain.working_memory.decay == 0.0  # 0.2 before 0.82.0
    # the projection that reads the carried interpretation back: 12 before 0.82.0
    assert brain.describe()["layout"]["composition"]["working_memory_scale"] == 1.2


def test_attention_is_an_explicit_learned_code_recall_recipe(tmp_path):
    brain = cd.Brain.compose(12, 2, modules=(24,), learned_code=8, attention=True, seed=3)
    assert brain.attention is True
    assert brain.learned_code is not None and brain.learned_code.units == 8
    assert brain.recall is not None
    assert brain.recall.window == 2
    assert brain.recall.next is True
    assert brain.recall.address == 0.0
    assert brain.recall.recency == 0.0
    assert brain.recall.temperature == 0.1
    assert brain.memory_choice is False
    description = brain.describe()
    assert description["genes"]["attention"] is True
    assert description["layout"]["composition"]["attention"] is True

    restored = cd.Brain.load(brain.save(tmp_path / "attention"))
    assert restored.attention is True
    assert restored.describe() == description


@pytest.mark.parametrize(
    "options",
    [
        {"attention": 1, "learned_code": 8},
        {"attention": True},
        {"attention": True, "learned_code": 8, "recall": False},
        {"attention": True, "learned_code": 8, "recall_window": 1},
        {"attention": True, "learned_code": 8, "recall_next": False},
    ],
)
def test_attention_refuses_incomplete_or_conflicting_recipes(options):
    with pytest.raises(ValueError):
        cd.Brain.compose(12, 2, modules=(24,), **options)


def test_attention_allows_expert_recall_tuning_without_changing_plain_defaults():
    tuned = cd.Brain.compose(
        12,
        2,
        modules=(24,),
        learned_code=8,
        attention=True,
        recall_window=3,
        recall_temperature=0.2,
        recall_recency=0.1,
    )
    assert tuned.recall is not None
    assert tuned.recall.window == 3
    assert tuned.recall.temperature == 0.2
    assert tuned.recall.recency == 0.1

    plain = cd.Brain.compose(12, 2, modules=(24,), learned_code=8)
    assert plain.attention is False and plain.recall is None
    assert "attention" not in plain.describe()["genes"]

    combined = cd.Brain.compose(
        12, 2, modules=(24,), learned_code=8, attention=True, memory_choice=True
    )
    assert combined.recall is not None
    assert combined.hippocampus is not None and combined._episodic is not None


@pytest.mark.parametrize("arousal", [True, {}, {"need": 2.0}])
def test_arousal_shorthand_uses_semantic_limits_not_mutation_search_bounds(arousal):
    brain = small(arousal=arousal)
    expected = cd.ArousalConfig(**arousal) if isinstance(arousal, dict) else cd.ArousalConfig()
    assert brain.arousal.config == expected
    assert brain.live(OBSERVATIONS[[0]]).shape == (1,)


def test_public_overrides_keep_teaching_and_reward_rates_distinct():
    learning = cd.LearnerConfig(eta=0.02, eta_bias=0.002, momentum=0.6)
    reward = cd.ActorCriticConfig(eta=0.04, eta_bias=0.004, gamma=0.7)
    brain = small(
        learning=learning,
        reward=reward,
        temperature=0.4,
        actor_eta=0.01,
        actor_eta_bias=None,
        arousal_need=2.0,
    )
    assert brain.learner.config == replace(learning, temperature=0.4)
    assert brain.basal_ganglia.config == replace(reward, eta=0.01, eta_bias=None)
    assert brain.arousal.config.need == 2.0
    assert learning.temperature == 0.2 and reward.eta == 0.04


@pytest.mark.parametrize(
    ("name", "value"),
    [("learning_beta", 0.2), ("learning_momentum", 0.5), ("actor_dopamine_cap", 0.0),
     ("actor_eligibility_steps", 4), ("arousal_need_scale", 0.0), ("memory_decay", 0.5),
     ("memory_replay", 2), ("memory_demonstrations", 0.5), ("memory_winners", 32),
     ("history_scale", 12.0), ("recall_scale", 12.0)],
)
def test_the_constants_of_the_composed_brain_are_refused_by_name(name, value):
    """The names 0.93.0 made constants are refused at compose and retune with the release and
    the constant, whatever the value, and describe emits none of them."""
    with pytest.raises(ValueError, match=f"{name} was retired in 0.93.0"):
        cd.Brain.compose(3, 2, modules=(8,), arousal=True, memory_choice=True, recall=True,
                         history_decays=(0.5,), **{name: value})
    brain = cd.Brain.compose(3, 2, modules=(8,), arousal=True, memory_choice=True, recall=True,
                             history_decays=(0.5,))
    with pytest.raises(ValueError, match=f"{name} was retired in 0.93.0"):
        brain.retune(**{name: value})
    assert name not in brain.describe()["genes"]


def test_a_whole_configuration_keeps_its_values_and_a_moved_constant_is_described(tmp_path):
    """``learning=``, ``reward=`` and ``arousal=`` are the expert route and keep their fields,
    as the chambers' frozen recipes compose; a mapping is the named route and refuses a
    constant. ``describe()`` names a constant only where an expert configuration moved it, by
    its old name, and a checkpoint carries and loads the expert's value."""
    expert = cd.Brain.compose(
        3, 2, modules=(8,), seed=1, learning=cd.LearnerConfig(beta=0.2, momentum=0.5),
        reward=cd.ActorCriticConfig(dopamine_cap=0.0, eligibility_steps=4),
        arousal=cd.ArousalConfig(need_scale=0.0),
    )
    config, actor = expert.learner.config, expert.basal_ganglia.config
    assert (config.beta, config.momentum, actor.dopamine_cap, actor.eligibility_steps) == (
        0.2, 0.5, 0.0, 4)
    assert expert.arousal.config.need_scale == 0.0
    genes = expert.describe()["genes"]
    assert (genes["learning_beta"], genes["learning_momentum"]) == (0.2, 0.5)
    assert (genes["actor_dopamine_cap"], genes["actor_eligibility_steps"]) == (0.0, 4)
    assert genes["arousal_need_scale"] == 0.0
    twin = cd.Brain.load(expert.save(tmp_path / "expert.npz"))
    assert twin.describe() == expert.describe()
    assert twin.learner.config == config and twin.basal_ganglia.config == actor
    founder = cd.Brain.compose(3, 2, modules=(8,), seed=1, arousal=True)
    assert not {"learning_beta", "learning_momentum", "actor_dopamine_cap",
                "actor_eligibility_steps", "arousal_need_scale"} & set(founder.describe()["genes"])
    with pytest.raises(ValueError, match="arousal_need_scale was retired in 0.93.0"):
        cd.Brain.compose(3, 2, modules=(8,), arousal={"need_scale": 0.5})
    with pytest.raises(ValueError, match="arousal_need_scale was retired in 0.93.0"):
        founder.retune(arousal={"need_scale": 0.0})
    # a standalone actor's eligibility_steps=None keeps its meaning, the learner's budget
    budget = cd.Brain.compose(3, 2, modules=(8,), reward=cd.ActorCriticConfig(eta=0.1))
    assert budget.basal_ganglia.config.eligibility_steps is None
    assert budget.describe()["genes"]["actor_eligibility_steps"] is None


def test_a_checkpoint_carries_each_constant_at_its_founder_and_another_value_is_refused(
    tmp_path,
):
    """The writer keeps emitting every key 0.92.1 wrote for a founder brain, at the constant.
    A saved value other than the constant is refused by name where the composed brain alone
    sets it (the episodic records' replay and demonstrations, the composition's scales); the
    expert route's fields load as saved, and a saved memory's decay and separation are its
    own."""
    brain = cd.Brain.compose(3, 2, modules=(8,), seed=2, arousal=True, memory_choice=True,
                             recall=True, history_decays=(0.5,))
    path = brain.save(tmp_path / "brain.npz")
    with np.load(path, allow_pickle=False) as saved:
        arrays = {name: saved[name].copy() for name in saved.files}
    meta, learner = json.loads(str(arrays["brain"])), json.loads(str(arrays["meta"]))
    assert (learner["config"]["beta"], learner["config"]["momentum"]) == (0.1, 0.9)
    assert (meta["actor"]["dopamine_cap"], meta["actor"]["eligibility_steps"]) == (1.0, 12)
    assert meta["arousal"]["config"]["need_scale"] == 0.1
    assert meta["memories"]["hippocampus"]["decay"] == 0.9
    assert meta["memories"]["hippocampus"]["separator"]["winners"] == 24
    assert meta["memories"]["episodic"]["replay"] == 8
    assert meta["memories"]["episodic"]["demonstrations"] == 1.0
    assert (meta["composition"]["history_scale"], meta["composition"]["recall_scale"]) == (
        1.2, 1.2)

    def saved_with(change, field):
        meta_, learner_ = json.loads(str(arrays["brain"])), json.loads(str(arrays["meta"]))
        change(meta_, learner_)
        bad = tmp_path / f"{field}.npz"
        np.savez(bad, **{**arrays, "brain": np.array(json.dumps(meta_)),
                         "meta": np.array(json.dumps(learner_))})
        with pytest.raises(ValueError, match=f"{field}=.*retired in 0.93.0"):
            cd.Brain.load(bad)

    saved_with(lambda m, learner: m["memories"]["episodic"].update(replay=2), "replay")
    saved_with(lambda m, learner: m["memories"]["episodic"].update(demonstrations=0.5),
               "demonstrations")
    saved_with(lambda m, learner: m["composition"].update(history_scale=12.0), "history_scale")
    saved_with(lambda m, learner: m["composition"].update(recall_scale=12.0), "recall_scale")
    # the expert route's fields and a memory's own decay and separation load as saved
    meta_, learner_ = json.loads(str(arrays["brain"])), json.loads(str(arrays["meta"]))
    learner_["config"].update(beta=0.2, momentum=0.5)
    meta_["actor"].update(dopamine_cap=0.0, eligibility_steps=4)
    meta_["arousal"]["config"].update(need_scale=0.0)
    meta_["memories"]["hippocampus"].update(decay=0.5)
    own = tmp_path / "own.npz"
    np.savez(own, **{**arrays, "brain": np.array(json.dumps(meta_)),
                     "meta": np.array(json.dumps(learner_))})
    loaded = cd.Brain.load(own)
    assert (loaded.learner.config.beta, loaded.learner.config.momentum) == (0.2, 0.5)
    assert loaded.basal_ganglia.config.eligibility_steps == 4
    assert loaded.arousal.config.need_scale == 0.0 and loaded.hippocampus.decay == 0.5


def test_the_sensory_scale_changes_only_the_sensory_projection():
    control = cd.Brain.compose(3, 5, modules=(9, 6), slots=(2, 3), seed=11)
    scaled = cd.Brain.compose(3, 5, modules=(9, 6), slots=(2, 3), seed=11, scales={"sensory": 4})
    first, second = control.connectome, scaled.connectome
    for field in ("pre", "post", "count"):
        np.testing.assert_array_equal(getattr(first, field), getattr(second, field))
    incoming = np.isin(first.pre, control.sensory_index)
    np.testing.assert_array_equal(second.sign[incoming], 4 * first.sign[incoming])
    np.testing.assert_array_equal(second.sign[~incoming], first.sign[~incoming])
    np.testing.assert_array_equal(control.brain.bias, scaled.brain.bias)
    np.testing.assert_array_equal(
        scaled.brain.efficacy[incoming], 4 * control.brain.efficacy[incoming]
    )
    np.testing.assert_array_equal(
        scaled.brain.efficacy[~incoming], control.brain.efficacy[~incoming]
    )


def test_multi_module_composition_description_survives_sorted_checkpoint_metadata(tmp_path):
    brain = cd.Brain.compose(
        3, 5, modules=(9, 6), observers=(4, 3), slots=(2, 3), scales={"sensory": 2.0}
    )
    restored = cd.Brain.load(brain.save(tmp_path / "multi-module"))
    assert restored.describe() == brain.describe()
    assert restored.describe()["layout"]["composition"]["modules"] == [9, 6]


@pytest.mark.parametrize(
    "options",
    [
        {"eta": 0.01},
        {"actor_ette": 0.01},
        {"actor_eta": True},
        {"learning_beta": np.nan},
        {"learning_eta": 0.1},
        {"learning_normalize": 0.99},
        {"teaching": "fast"},
        {"teaching": "normalized", "learning": cd.LearnerConfig()},
        {"arousal_need": np.inf},
        {"learning_free_steps": 2.5},
        {"actor_center_scale": True},
        {"actor_dopamine_center": 0.0},
        {"learning_temperature": 0.4},
        {"arousal": {"need": 1.0}, "arousal_need": 2.0},
        {"sensory_scale": 4.0},
        {"scales": {"sensory": -1.0}},
        {"scales": {"sensory": True}},
        {"scales": {"cortex": 1.0}},
        {"scales": 1.0},
    ],
)
def test_constructor_refuses_unknown_conflicting_and_invalid_settings(options):
    with pytest.raises((ValueError, TypeError)):
        small(**options)


@pytest.mark.parametrize(
    "invalid",
    [
        {"actor_eta": -1.0},
        {"arousal_need": np.inf},
        {"learning_free_steps": True},
        {"memory_rate": 2.0},
        {"working_memory_decay": 1.0},
        {"working_memory_focus": -1.0},
        {"surprise_threshold": 0.1},
        {"sensory_scale": 4.0},
        {"scales": {"sensory": 4.0}},
        {"learning_eta": 0.1},
        {"teaching": "slow"},
        {"resting_bias": 0.1},
        {"reset_arousal": 1},
        {"learning_beta": 0.2},
    ],
)
def test_invalid_retune_is_atomic_with_acquired_state_and_pending_reward(tmp_path, invalid):
    brain = small()
    acquire(brain)
    before = archive(brain, tmp_path, "before")
    settings = {"temperature": 0.35, "consolidation": 0.25}
    settings.update(invalid)
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        brain.retune(**settings)
    assert_archives_equal(before, archive(brain, tmp_path, "after"))
    assert brain.pending_feedback


def test_retune_matches_the_existing_stage_operation_and_preserves_pending_credit(tmp_path):
    brain = small(efference_amplitude=0.2)
    acquire(brain)
    path = brain.save(tmp_path / "acquired")
    legacy = cd.Brain.load(path)
    before = archive(brain, tmp_path, "before-stage")
    pending = brain.basal_ganglia._pending
    assert (
        brain.retune(
            arousal={"need": 0.0, "heat": 0.0},
            temperature=0.3,
            actor_eta=0.003,
            actor_eta_bias=0.0003,
        )
        is brain
    )
    brain.arousal.reset()
    assert brain.basal_ganglia._pending is pending
    after = archive(brain, tmp_path, "after-stage")
    for name in before:
        if name not in ("generic", "meta", "brain"):
            np.testing.assert_array_equal(before[name], after[name], err_msg=name)
    legacy.arousal.config = replace(legacy.arousal.config, need=0.0, heat=0.0)
    legacy.learner.config = replace(legacy.learner.config, temperature=0.3)
    legacy.basal_ganglia.config = replace(legacy.basal_ganglia.config, eta=0.003, eta_bias=0.0003)
    legacy.arousal.reset()
    restored = cd.Brain.load(brain.save(tmp_path / "retuned"))
    assert restored.describe() == brain.describe()
    for index, reward in enumerate((0.7, -0.2, 0.4, 0.0)):
        observation = OBSERVATIONS[[index % 3]]
        expected = legacy.live(observation, reward=[reward])
        np.testing.assert_array_equal(brain.live(observation, reward=[reward]), expected)
        np.testing.assert_array_equal(restored.live(observation, reward=[reward]), expected)
        np.testing.assert_array_equal(brain.brain.efficacy, legacy.brain.efficacy)
        np.testing.assert_array_equal(brain.brain.bias, legacy.brain.bias)
        np.testing.assert_array_equal(
            brain.hippocampus.consolidated, legacy.hippocampus.consolidated
        )
    assert_archives_equal(archive(brain, tmp_path, "live"), archive(restored, tmp_path, "resumed"))


def test_memory_and_trace_tuning_keeps_learned_contents(tmp_path):
    brain = small(efference_amplitude=0.2)
    acquire(brain)
    before = archive(brain, tmp_path, "before-memory")
    brain.retune(
        memory_rate=0.7,
        memory_amplitude=1.2,
        consolidation=0.2,
        working_memory_decay=0.4,
        working_memory_amplitude=0.2,
        efference_decay=0.4,
        efference_amplitude=0.1,
    )
    after = archive(brain, tmp_path, "after-memory")
    for name in before:
        if name not in ("generic", "meta", "brain"):
            np.testing.assert_array_equal(before[name], after[name], err_msg=name)
    restored = cd.Brain.load(brain.save(tmp_path / "memory-tuned"))
    assert brain.describe() == restored.describe()
    np.testing.assert_array_equal(
        brain.live(OBSERVATIONS[[2]], reward=[0.2]),
        restored.live(OBSERVATIONS[[2]], reward=[0.2]),
    )


@pytest.mark.parametrize(
    "options",
    [
        {"arousal_need": 1.0},
        {"arousal": {"need": 1.0}},
        {"reset_arousal": True},
        {"memory_amplitude": 2.0},
        {"consolidation": 0.2},
        {"efference_amplitude": 0.2},
    ],
)
def test_retune_cannot_create_missing_components(options):
    brain = cd.Brain.compose(3, 2, modules=(8,), episodic=False)
    with pytest.raises((ValueError, TypeError)):
        brain.retune(**options)


def test_description_is_detached_observational_and_survives_checkpoints(tmp_path):
    brain = small(scales={"sensory": 2.0})
    description = brain.describe()
    json.dumps(description, allow_nan=False)
    assert description["genes"]["actor_eta"] == 0.03
    assert description["genes"]["teaching"] == "founder"
    description["genes"]["actor_eta"] = 999
    assert brain.basal_ganglia.config.eta == 0.03
    acquire(brain)
    before = archive(brain, tmp_path, "before-description")
    brain.describe()
    assert_archives_equal(before, archive(brain, tmp_path, "after-description"))
    restored = cd.Brain.load(brain.save(tmp_path / "described"))
    assert brain.describe() == restored.describe()


def test_pending_feedback_covers_sampled_routine_and_consumed_outcomes():
    brain = small()
    assert not brain.pending_feedback
    brain.live(OBSERVATIONS[[0]])
    assert brain.pending_feedback
    brain.learn([0.2], [False], OBSERVATIONS[[1]])
    assert not brain.pending_feedback
    brain.reset()
    brain.retune(arousal={"youth": 0, "threshold": 10.0})
    brain.live(OBSERVATIONS[[0]])
    assert brain.last_arousal["mode"] == "routine" and brain.pending_feedback
    brain.act(OBSERVATIONS[[1]], greedy=True)
    assert not brain.pending_feedback
    brain.reset()
    assert not brain.pending_feedback


def test_the_trace_focus_of_0843_is_refused_by_name_and_read_at_zero(tmp_path):
    """The working trace's focus, a weighting of each neuron by its movement, is removed: the
    name is refused at compose and retune, and a checkpoint of 0.84.3 loads at the founder
    focus of zero and is refused above it, each refusal naming 0.84.3."""
    with pytest.raises(ValueError, match="working_memory_focus was retired.*0.84.3"):
        small(working_memory_focus=0.5)
    brain = small()
    with pytest.raises(ValueError, match="working_memory_focus was retired.*0.84.3"):
        brain.retune(working_memory_focus=0.0)
    assert "working_memory_focus" not in brain.describe()["genes"]
    path = brain.save(tmp_path / "trace.npz")
    with np.load(path, allow_pickle=False) as saved:
        arrays = earlier({name: saved[name].copy() for name in saved.files})
    meta = metadata(arrays)
    for focus, loads in ((0.0, True), (0.5, False)):
        meta["working_memory"]["focus"] = focus
        arrays["generic"] = np.array(json.dumps(meta))
        np.savez(path, **arrays)
        if loads:
            assert cd.Brain.load(path).working_memory.to_dict() == brain.working_memory.to_dict()
        else:
            with pytest.raises(ValueError, match="focus=0.5.*retired in 0.85.0.*0.84.3"):
                cd.Brain.load(path)


def test_the_resting_bias_of_0843_is_refused_by_name_and_its_record_loads(tmp_path):
    """The composed brain's resting bias, an initial bias of the processing regions, is
    removed: the name is refused at compose and retune with a message naming 0.84.3, and the
    value a checkpoint of 0.84.3 recorded is provenance its saved biases already hold."""
    with pytest.raises(ValueError, match="resting_bias was retired.*0.84.3"):
        small(resting_bias=0.5)
    brain = small()
    with pytest.raises(ValueError, match="resting_bias was retired.*0.84.3"):
        brain.retune(resting_bias=0.5)
    assert "resting_bias" not in brain.describe()["layout"]["composition"]
    path = brain.save(tmp_path / "recorded.npz")
    with np.load(path, allow_pickle=False) as saved:
        arrays = earlier({name: saved[name].copy() for name in saved.files})
    meta = metadata(arrays)
    meta["resting_bias"] = 0.5
    arrays["generic"] = np.array(json.dumps(meta))
    np.savez(path, **arrays)
    np.testing.assert_array_equal(cd.Brain.load(path).brain.bias, brain.brain.bias)


def test_the_teaching_point_sets_the_lessons_steps_and_normalization_alone():
    founder = cd.Brain.compose(3, 2, modules=(8,), seed=1)
    named = cd.Brain.compose(3, 2, modules=(8,), seed=1, teaching="founder")
    assert named.learner.config == founder.learner.config  # the founder by its name
    normalized = cd.Brain.compose(3, 2, modules=(8,), seed=1, teaching="normalized")
    config = normalized.learner.config
    assert (config.eta, config.eta_bias, config.normalize, config.normalize_floor) == (
        0.003, 0.0003, 0.99, 1e-4
    )
    assert config.momentum == 0.9
    assert replace(config, eta=0.5, eta_bias=0.05, normalize=0.0, normalize_floor=1e-3) == (
        founder.learner.config
    )  # nothing else moves
    founder.retune(teaching="normalized")
    assert founder.learner.config == config
    founder.retune(teaching="founder", learning_surprise=True)
    assert founder.learner.config == replace(named.learner.config, surprise=True)
    with pytest.raises(ValueError, match="teaching"):
        cd.Brain.compose(3, 2, learning_eta=0.1)
    with pytest.raises(ValueError, match="one of them"):
        founder.retune(teaching="normalized", learning=cd.LearnerConfig())


def test_scales_name_the_projection_magnitudes_and_keep_their_founders():
    founder = cd.Brain.compose(3, 2, modules=(8,), seed=4, history_decays=(0.5,), recall=True)
    named = cd.Brain.compose(
        3, 2, modules=(8,), seed=4, history_decays=(0.5,), recall=True,
        scales={"sensory": 1.0, "trace": 1.2},
    )
    np.testing.assert_array_equal(named.brain.efficacy, founder.brain.efficacy)
    composition = founder.describe()["layout"]["composition"]
    assert (
        composition["sensory_scale"], composition["working_memory_scale"],
        composition["history_scale"], composition["recall_scale"],
    ) == (1.0, 1.2, 1.2, 1.2)
    for old, key in (("sensory_scale", "sensory"), ("working_memory_scale", "trace")):
        with pytest.raises(ValueError, match=f"scales={{'{key}'"):
            cd.Brain.compose(3, 2, **{old: 1.0})
    # the history's and the recall memory's projection scales are constants since 0.93.0,
    # taken at the constant alone
    for key in ("history", "recall"):
        with pytest.raises(ValueError, match=f"{key}_scale was retired in 0.93.0"):
            cd.Brain.compose(3, 2, modules=(8,), scales={key: 12.0})
    at_constant = cd.Brain.compose(
        3, 2, modules=(8,), seed=4, history_decays=(0.5,), recall=True,
        scales={"history": 1.2, "recall": 1.2},
    )
    np.testing.assert_array_equal(at_constant.brain.efficacy, founder.brain.efficacy)


def test_the_description_has_four_sections_and_names_every_gene_as_compose_takes_it():
    brain = cd.Brain.compose(
        4, 2, modules=(8,), seed=2, arousal=True, recall=True, memory_choice=True,
        history_decays=(0.5,), efference_amplitude=1.0, scales={"trace": 2.0},
    )
    described = brain.describe()
    assert list(described) == ["layout", "genes", "memories", "pending"]
    assert described["layout"]["composition"]["working_memory_scale"] == 2.0
    assert described["layout"]["code"]["kind"] == "observation"
    assert list(described["memories"]) == list(brain.memories)
    assert described["memories"]["episodic"] == brain.memories["episodic"].to_dict()
    assert described["pending"] is False
    genes = described["genes"]
    assert genes["teaching"] == "founder"
    assert not {"learning_eta", "learning_eta_bias", "learning_normalize",
                "learning_normalize_floor", "learning_nudge"} & set(genes)
    # every runtime gene goes back into retune as it is
    brain.retune(**{k: v for k, v in genes.items() if not k.startswith(
        ("recall_capacity", "recall_next", "history_decays"))})
    assert brain.describe()["genes"] == genes
    brain.retune(teaching="normalized")
    assert brain.describe()["genes"]["teaching"] == "normalized"
    brain.retune(learning=replace(brain.learner.config, eta=0.2))
    assert brain.describe()["genes"]["teaching"] == "custom"


def test_fit_keeps_the_arousal_of_a_life_and_retune_resets_nothing():
    brain = cd.Brain.compose(4, 2, modules=(8,), seed=5, arousal={"youth": 3})
    x = np.eye(4)[[0]]
    brain.live(x)
    for t in range(8):
        brain.live(np.eye(4)[[t % 4]], reward=[float(t % 2)])
    state = brain.arousal.to_dict()
    brain.fit(np.eye(4), np.array([0, 1, 0, 1]), epochs=1, batch=4)
    assert brain.arousal.to_dict() == state  # a control on independent samples
    brain.retune(arousal_heat=1.5)
    assert {k: v for k, v in brain.arousal.to_dict().items() if k != "config"} == {
        k: v for k, v in state.items() if k != "config"
    }
    with pytest.raises(TypeError, match="unknown brain setting: 'reset_arousal'"):
        brain.retune(reset_arousal=True)
