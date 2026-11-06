"""Arousal and ``Brain.live``: routine changes nothing, surprise and want wake the brain."""

import copy
import json
from dataclasses import replace

import numpy as np
import pytest
from checkpoints import flat, metadata, write

import cadence as cd


def calm_brain(seed: int = 0, **genes) -> cd.Brain:
    """A brain born calm (no youth), so its first moments are routine."""
    return cd.Brain.compose(
        4,
        2,
        modules=(16,),
        seed=seed,
        working_memory_amplitude=0.0,
        arousal=cd.ArousalConfig(**{"youth": 0, **genes}),
    )


def snapshot(brain: cd.Brain) -> dict[str, np.ndarray]:
    agent, memory = brain.basal_ganglia, brain.hippocampus
    return {
        "efficacy": np.asarray(brain.brain.efficacy).copy(),
        "bias": np.asarray(brain.brain.bias).copy(),
        "critic": np.append(agent.w_critic, agent.b_critic),
        "consolidated": memory.consolidated.copy(),
        "strength": memory.strength.copy(),
        "counts": np.array([brain.learner.updates, agent.updates, memory.writes]),
    }


# -- the law


def test_arousal_genes_are_validated_and_declare_their_space():
    config = cd.ArousalConfig()
    # need_scale is a gene of the law and not of the space, so the founder's seeded
    # mutations draw as they did before it
    assert set(config.to_dict()) == set(cd.ArousalConfig.space()) | {"need_scale"}
    assert cd.ArousalConfig(**config.to_dict()) == config
    for bad in (
        {"threshold": -0.1},
        {"decay": 1.0},
        {"tolerance": float("nan")},
        {"floor": -1.0},
        {"fast": 0.001, "slow": 0.01},
        {"slow": 0.0},
        {"heat": -1.0},
        {"youth": 1.5},
        {"youth": True},
    ):
        with pytest.raises(ValueError):
            cd.ArousalConfig(**bad)
    mutate = cd.genes(cd.ArousalConfig.space())
    child = mutate(config.to_dict(), np.random.default_rng(0))
    assert set(child) == set(config.to_dict())


def test_a_newborn_is_aroused_for_its_youth_and_a_first_outcome_is_no_surprise():
    arousal = cd.Arousal(cd.ArousalConfig(youth=3))
    assert arousal.aroused and arousal.mode == "aroused"
    surprise, want = arousal.outcome(5.0, 0.0)
    assert surprise == 0.0 and want == 0.0
    for _ in range(3):
        arousal.lived(1)
    assert not arousal.aroused and arousal.moments == {"routine": 0, "aroused": 3}


def test_a_usual_world_stays_calm_and_a_contradiction_wakes_then_settles():
    rng = np.random.default_rng(0)
    arousal = cd.Arousal(cd.ArousalConfig(youth=0))
    modes = []
    for _ in range(1500):
        arousal.outcome(0.3 * rng.random(), float(rng.random() < 0.5))
        modes.append(arousal.aroused)
        arousal.lived(1)
    assert np.mean(modes[500:]) < 0.05  # the usual error and reward are nothing
    surprise, _ = arousal.outcome(3.0, 0.0)  # ten times the usual error
    assert surprise > 1.0 and arousal.aroused
    for _ in range(60):
        arousal.outcome(0.3 * rng.random(), float(rng.random() < 0.5))
    assert not arousal.aroused  # one contradiction is over within a few dozen moments


def test_reward_below_the_long_run_is_a_want_that_habituates():
    arousal = cd.Arousal(cd.ArousalConfig(youth=0))
    for moment in range(200):  # a varied youth sets the spread of reward: 0, 1, 0, 1
        arousal.outcome(0.0, float(moment % 2))
    for _ in range(600):  # then a calm life at its usual pay
        arousal.outcome(0.0, 0.5, learned=False)
    assert arousal.want == pytest.approx(0.0, abs=1e-3) and not arousal.aroused
    aroused = []
    for _ in range(1500):  # the pay halves; the forecast is right every time
        surprise, _ = arousal.outcome(0.0, 0.25, learned=False)
        assert surprise == 0.0
        aroused.append(arousal.aroused)
    assert all(aroused[30:120])  # no surprise, and still roused: the reward is missing
    assert not any(aroused[-300:]) and arousal.want < 0.05  # the poorer life becomes usual
    assert arousal.heat >= 1.0


def test_a_need_is_a_want_of_its_own_that_a_sparse_reward_meets_and_that_never_habituates():
    """Fed once in fourteen moments, a brain's income is 1/14 per moment while its reward
    spread is near 1/sqrt(14): losing the food leaves the long-run want below the
    threshold, and it habituates; the need's want is the unmet share of the need."""
    # the law without the scaled need (need_scale 0, the law of 0.84.3): the need alone
    content = cd.Arousal(cd.ArousalConfig(youth=0, need_scale=0.0))
    needy = cd.Arousal(cd.ArousalConfig(youth=0, need=0.04, need_scale=0.0))
    assert needy.want == 1.0 and needy.heat == 3.0  # never paid: the whole need is unmet
    assert content.want == 0.0
    peaks = {}
    for arousal in (content, needy):
        for moment in range(2800):  # a youth that learns, then the same pay in routine
            young = moment < 1400
            arousal.outcome(0.1 if young else 0.0, float(moment % 14 == 0), learned=young)
        assert arousal.want < 0.1 and not arousal.aroused  # the need is met by the income
        wants = []
        for _ in range(100):  # the food stops; every forecast of nothing is right
            arousal.outcome(0.0, 0.0, learned=False)
            wants.append(arousal.want)
        peaks[arousal] = max(wants)
    assert peaks[content] < 1 / np.sqrt(14) and not content.aroused  # below the threshold
    assert needy.want > 0.95 and needy.heat > 2.9 and needy.aroused
    for _ in range(3000):
        content.outcome(0.0, 0.0, learned=False)
        needy.outcome(0.0, 0.0, learned=False)
    assert content.want < 0.05 and not content.aroused  # the poorer life became usual
    assert needy.want == 1.0 and needy.aroused  # a need is a level: it stays unmet
    scaled = cd.Arousal(cd.ArousalConfig(youth=0, need=40 * 0.04, need_scale=0.0))
    plain = cd.Arousal(cd.ArousalConfig(youth=0, need=0.04, need_scale=0.0))
    for moment in range(300):
        a = plain.outcome(0.1, float(moment % 14 == 0), learned=moment < 200)
        b = scaled.outcome(4.0, 40.0 * float(moment % 14 == 0), learned=moment < 200)
        assert a == pytest.approx(b, rel=1e-9, abs=1e-12)
    with pytest.raises(ValueError):
        cd.ArousalConfig(need=-0.1)
    assert "need" in cd.ArousalConfig.space() and cd.ArousalConfig().need == 0.0


@pytest.mark.parametrize("need", [-0.1, True, np.bool_(False), np.nan, np.inf, -np.inf, "0.1"])
def test_need_rejects_invalid_genes(need):
    with pytest.raises(ValueError, match="need"):
        cd.ArousalConfig(need=need)


@pytest.mark.parametrize(
    "need", [float(np.nextafter(0.0, 1.0)), np.nextafter(0.0, 1.0), np.float32(3e38)]
)
def test_need_clips_extreme_shortfalls_without_overflow(need):
    arousal = cd.Arousal(cd.ArousalConfig(youth=0, need=need, fast=1.0, slow=1.0))
    with np.errstate(all="raise"):
        for reward, expected in [(-1e38, 1.0), (0.0, 1.0), (float(need), 0.0)]:
            assert arousal.outcome(0.0, reward) == (0.0, expected)
            assert arousal.want == expected
            assert np.isfinite(arousal.level) and np.isfinite(arousal.heat)
    assert cd.Arousal.from_dict(arousal.to_dict()).to_dict() == arousal.to_dict()


def test_every_outcome_changes_whether_need_is_met_even_with_equal_rates():
    arousal = cd.Arousal(cd.ArousalConfig(youth=0, need=0.25, fast=1.0, slow=1.0))
    assert arousal.outcome(0.0, 0.25) == (0.0, 0.0)
    assert arousal.outcome(0.0, 0.0, own=False) == (0.0, 1.0)
    assert arousal.recent == 0.0
    assert arousal.outcome(0.0, 0.0) == (0.0, 1.0)
    assert arousal.outcome(0.0, 0.5, own=False) == (0.0, 0.0)
    assert arousal.recent == 0.5
    assert arousal.outcome(0.0, 0.5) == (0.0, 0.0)


def test_the_law_is_unchanged_by_the_scale_and_the_zero_of_reward():
    # a shift of the rewards preserves the law where the need and the need scale are zero
    rng = np.random.default_rng(3)
    plain = cd.Arousal(cd.ArousalConfig(youth=0, need_scale=0.0))
    moved = cd.Arousal(cd.ArousalConfig(youth=0, need_scale=0.0))
    for moment in range(600):
        reward = float(rng.random() < (0.7 if moment < 400 else 0.2))
        error = float(0.05 + 0.3 * rng.random() + (moment == 400))
        own, learned = bool(rng.random() < 0.9), bool(rng.random() < 0.3)
        a = plain.outcome(error, reward, own=own, learned=learned)
        b = moved.outcome(40.0 * error, 40.0 * reward - 7.0, own=own, learned=learned)
        assert a == pytest.approx(b, rel=1e-9, abs=1e-12)
        assert plain.level == pytest.approx(moved.level, rel=1e-9, abs=1e-12)
        assert plain.aroused == moved.aroused
    assert moved.scale == pytest.approx(40.0 * plain.scale)


def test_explored_outcomes_change_income_without_surprising_or_recalibrating_error():
    arousal = cd.Arousal(cd.ArousalConfig(youth=0))
    for moment in range(300):
        arousal.outcome(0.1, float(moment % 2))
    usual = (arousal.usual, arousal.outcomes)
    income = (arousal.recent, arousal.longrun)
    surprise, _ = arousal.outcome(5.0, -3.0, own=False)  # a costly exploration
    assert surprise == 0.0
    assert (arousal.usual, arousal.outcomes) == usual
    assert arousal.recent < income[0] and arousal.longrun < income[1]
    assert arousal.rewards == 301
    assert arousal.outcome(5.0, -3.0)[0] > 1.0  # the same outcome of its own best guess


def test_routine_outcomes_leave_the_reward_scale_alone():
    arousal = cd.Arousal(cd.ArousalConfig(youth=0))
    for moment in range(200):
        arousal.outcome(0.1, float(moment % 2))  # learned from: the scale forms
    scale, spreads = arousal.scale, arousal.spreads
    assert scale == pytest.approx(0.5, abs=0.05) and spreads == 200
    for _ in range(3000):
        arousal.outcome(0.0, 0.5, learned=False)  # a long calm
    assert arousal.scale == scale and arousal.spreads == spreads
    arousal.outcome(4.0, -2.0, learned=False)  # the outcome that wakes the brain enters
    assert arousal.aroused and arousal.spreads == spreads + 1


def test_equal_rates_remove_the_want_and_a_wide_tolerance_removes_surprise():
    no_want = cd.Arousal(cd.ArousalConfig(youth=0, fast=0.005, slow=0.005))
    no_surprise = cd.Arousal(cd.ArousalConfig(youth=0, tolerance=20.0, floor=1.0))
    for arousal in (no_want, no_surprise):
        for _ in range(300):
            arousal.outcome(0.1, 1.0)
    for _ in range(50):
        no_want.outcome(0.1, 0.0)
    assert no_want.want == 0.0 and not no_want.aroused
    assert no_surprise.outcome(1.5, 1.0)[0] == 0.0


def test_the_record_channel_of_0843_is_refused_by_name():
    """The record channel, a second surprise at the outcome the action's record forecast, is
    removed: its gene is refused by name at compose, retune and in an arousal mapping, each
    refusal naming 0.84.3, and a saved state carries none of its readings."""
    with pytest.raises(TypeError):
        cd.ArousalConfig(record_surprise=0.0)
    for genes in ({"arousal": True, "arousal_record_surprise": 0.0},
                  {"arousal": {"record_surprise": 0.0}}):
        with pytest.raises(ValueError, match="record_surprise was retired.*0.84.3"):
            cd.Brain.compose(4, 2, modules=(8,), **genes)
    brain = calm_brain()
    with pytest.raises(ValueError, match="arousal_record_surprise was retired.*0.84.3"):
        brain.retune(arousal_record_surprise=1.0)
    assert "arousal_record_surprise" not in brain.describe()["genes"]
    assert "record_surprise" not in cd.ArousalConfig.space()
    state = cd.Arousal(cd.ArousalConfig(youth=2)).to_dict()
    assert state["format"] == "cadence-arousal/4"
    assert "records" not in state and "usual_record" not in state


def test_arousal_state_round_trips_and_rejects_corruption():
    arousal = cd.Arousal(cd.ArousalConfig(youth=2))
    for value in (0.5, 1.0, 0.2):
        arousal.outcome(value, value)
        arousal.lived(7, 3)
    saved = arousal.to_dict()
    assert cd.Arousal.from_dict(saved).to_dict() == saved
    for key, bad in (("level", -1.0), ("age", -1), ("usual", float("nan")), ("moments", {})):
        with pytest.raises(ValueError):
            cd.Arousal.from_dict({**saved, key: bad})
    with pytest.raises(ValueError):
        cd.Arousal.from_dict({k: v for k, v in saved.items() if k != "recent"})
    with pytest.raises(ValueError):
        arousal.outcome(float("inf"), 0.0)


def test_an_arousal_of_another_format_is_refused_naming_the_release_that_loads_it():
    """The formats of 0.84.3 and earlier leave: cadence 0.94.0 loads them and saves the one this
    version reads. The format of a positive hindsight keeps its own sentence, and a save of the
    format this version reads needs every gene."""
    arousal = cd.Arousal(cd.ArousalConfig(youth=2, need_scale=0.0))
    for value in (0.5, 1.0, 0.2):
        arousal.outcome(value, value)
        arousal.lived(7, 3)
    state = arousal.to_dict()
    assert cd.Arousal.from_dict(state).to_dict() == state
    for format in (None, "cadence-arousal/0", "cadence-arousal/1", "cadence-arousal/2",
                   "cadence-arousal/3"):
        with pytest.raises(ValueError, match="0.84.3 and earlier wrote: cadence 0.94.0 loads it"):
            cd.Arousal.from_dict({**state, "format": format})
    with pytest.raises(ValueError, match="0.84.3 and earlier wrote: cadence 0.94.0 loads it"):
        cd.Arousal.from_dict({k: v for k, v in state.items() if k != "format"})
    # the format of a positive hindsight (0.88.1 to 0.92.1), retired in 0.93.0
    with pytest.raises(ValueError, match="cadence-arousal/5.*retired in 0.93.0"):
        cd.Arousal.from_dict({**state, "format": "cadence-arousal/5"})
    for gene in state["config"]:
        corrupt = {**state, "config": {k: v for k, v in state["config"].items() if k != gene}}
        with pytest.raises(ValueError, match="incomplete saved arousal config"):
            cd.Arousal.from_dict(corrupt)


# -- the behaviour temperature


def test_a_behaviour_temperature_widens_sampling_and_leaves_the_policy_alone():
    brain = cd.Brain.compose(4, 3, modules=(8,), seed=0)
    x = np.eye(4)[[0]]
    brain.act(x, greedy=True)
    agent = brain.basal_ganglia
    own = agent.probabilities(agent.state)
    hot = agent.probabilities(agent.state, 5.0)
    assert np.ptp(hot) < np.ptp(own)  # flatter at a higher temperature
    np.testing.assert_allclose(agent.probabilities(agent.state, 0.2), own)
    twin = copy.deepcopy(brain)  # the same stream: the greedy act reads no temperature
    np.testing.assert_array_equal(
        brain.act(x, greedy=True), twin.act(x, greedy=True, temperature=9.0)
    )
    before = agent.rng.bit_generator.state
    for bad in (0.0, -1.0, float("nan"), True, "hot"):
        with pytest.raises(ValueError):
            brain.act(x, temperature=bad)
    assert agent.rng.bit_generator.state == before  # an invalid call drew nothing
    action = brain.act(x, temperature=5.0)
    assert action.shape == (1,) and agent._pending is not None


# -- live


def test_live_needs_arousal_one_stream_and_a_preceding_action():
    plain = cd.Brain.compose(4, 2, modules=(8,), seed=0)
    with pytest.raises(ValueError, match="arousal"):
        plain.live(np.eye(4)[[0]])
    with pytest.raises(ValueError, match="threshold"):
        cd.Brain.compose(4, 2, modules=(8,), arousal={"threshold": -0.2})
    brain = calm_brain()
    with pytest.raises(ValueError, match="one continuing stream"):
        brain.live(np.eye(4)[:2])
    with pytest.raises(RuntimeError, match="preceding action"):
        brain.live(np.eye(4)[[0]], reward=[1.0])
    brain.live(np.eye(4)[[0]])
    for bad in ([1.0, 2.0], [float("nan")], 1.0):
        with pytest.raises(ValueError):
            brain.live(np.eye(4)[[1]], reward=bad)
    with pytest.raises(ValueError):
        brain.live(np.eye(4)[[1]], reward=[0.0], done=[0])


def test_routine_answers_greedily_and_changes_nothing():
    brain = calm_brain()
    eye = np.eye(4)
    brain.live(eye[[0]])
    before = snapshot(brain)
    for moment in range(12):
        x = eye[[moment % 4]]
        action = brain.live(x, reward=[0.0])
        reading = brain.last_arousal
        assert reading["mode"] == "routine" and not reading["learned"] and not reading["recorded"]
        assert reading["temperature"] is None and reading["learning_sweeps"] == 0
        assert brain.basal_ganglia._pending is None and brain.last_learning == {}
        motor = brain.basal_ganglia.state.activation[0, brain.motor_index]
        assert action[0] == int(np.argmax(motor))  # the settled state's own choice
    after = snapshot(brain)
    for name in before:
        np.testing.assert_array_equal(after[name], before[name])
    assert brain.arousal.moments == {"routine": 13, "aroused": 0}
    assert brain.arousal.sweeps["routine"] > 0 and brain.arousal.learning_sweeps == 0


def test_eligibility_fades_through_routine_moments_and_ends_with_the_episode():
    brain = calm_brain(seed=1, youth=6)
    eye = np.eye(4)
    agent = brain.basal_ganglia
    brain.live(eye[[0]])
    for moment in range(1, 7):  # a quiet youth: six sampled actions and their outcomes
        brain.live(eye[[moment % 4]], reward=[0.0])
    assert brain.last_arousal["mode"] == "routine" and brain.last_arousal["learned"]
    traces = {name: getattr(agent, name).copy() for name in ("trace", "trace_bias", "trace_critic")}
    assert all(np.abs(value).max() > 0 for value in traces.values())
    before = snapshot(brain)
    decay = agent.config.gamma * agent.config.lam
    for gap in range(1, 5):  # each routine moment fades the credit of the sampled actions once
        brain.live(eye[[gap % 4]], reward=[0.0])
        assert brain.last_arousal["mode"] == "routine" and not brain.last_arousal["learned"]
        for name, value in traces.items():
            np.testing.assert_allclose(getattr(agent, name), decay**gap * value, rtol=1e-12)
    for name, value in snapshot(brain).items():
        np.testing.assert_array_equal(value, before[name])  # and nothing is learned
    brain.live(eye[[1]], reward=[0.0], done=[True])  # the episode ends in routine
    assert not any(getattr(agent, name).any() for name in traces)


def test_fade_checks_its_rows_and_is_nothing_without_traces():
    brain = calm_brain(youth=3)
    agent = brain.basal_ganglia
    agent.fade()  # no eligibility yet
    agent.fade(np.array([True]))
    assert agent.trace is None and agent.trace_critic is None
    eye = np.eye(4)
    brain.live(eye[[0]])
    brain.live(eye[[1]], reward=[0.0])
    kept = agent.trace.copy()
    for bad in (np.array([1]), np.zeros((1, 1), bool), np.zeros(2, bool)):
        with pytest.raises(ValueError):
            agent.fade(bad)
    np.testing.assert_array_equal(agent.trace, kept)  # a refused call fades nothing
    agent.fade(np.array([False]))
    np.testing.assert_allclose(agent.trace, agent.config.gamma * agent.config.lam * kept)


def test_a_contradicting_outcome_wakes_the_brain_and_is_recorded_once():
    brain = calm_brain()
    eye = np.eye(4)
    first = brain.live(eye[[0]])
    for _ in range(5):
        brain.live(eye[[0]], reward=[0.0])
    lived_action = int(brain._lived[1][0])
    writes = brain.hippocampus.writes
    brain.live(eye[[1]], reward=[-1.0])  # the forecast was nothing; the outcome is a blow
    reading = brain.last_arousal
    assert reading["mode"] == "aroused" and reading["recorded"] and not reading["learned"]
    assert reading["surprise"] > 0 and reading["error"] == pytest.approx(1.0)
    assert reading["temperature"] >= brain.learner.config.temperature
    assert brain.hippocampus.writes == writes + 1
    recalled = brain.hippocampus.recall(eye[[0]])[0]
    assert recalled[lived_action] < -0.9 and recalled[1 - lived_action] == 0.0
    assert brain.basal_ganglia._pending is not None  # the waking action keeps its eligibility
    updates = brain.basal_ganglia.updates
    brain.live(eye[[2]], reward=[0.0])
    assert brain.last_arousal["learned"] and brain.basal_ganglia.updates == updates + 1
    assert brain.hippocampus.writes == writes + 2
    assert first.shape == (1,)


def test_every_outcome_is_taken_exactly_once_through_both_modes():
    brain = calm_brain(seed=3, youth=40)
    eye = np.eye(4)
    rng = np.random.default_rng(0)
    cue = 0
    action = brain.live(eye[[cue]])
    sampled = 0
    for moment in range(400):
        # the rewarded action follows the cue, and the rule turns over halfway: a life through
        # youth, routine and repair
        target = (cue % 2) ^ int(moment >= 200)
        reward = 1.0 if action[0] == target else -1.0
        was_sampled = brain.basal_ganglia._pending is not None
        sampled += was_sampled
        updates = brain.basal_ganglia.updates
        cue = int(rng.integers(4))
        action = brain.live(eye[[cue]], reward=[reward])
        assert brain.last_arousal["learned"] == was_sampled
        assert brain.basal_ganglia.updates == updates + int(was_sampled)
    assert brain.basal_ganglia.updates == sampled
    moments = brain.arousal.moments
    assert moments["routine"] > 100 and moments["aroused"] >= 40
    assert moments["routine"] + moments["aroused"] == 401 == brain.arousal.age


@pytest.mark.parametrize("wake", [False, True])
@pytest.mark.parametrize("need", [0.0, 0.15])
def test_a_saved_life_continues_identically(tmp_path, wake, need):
    eye = np.eye(4)

    def world(action: int, moment: int) -> tuple[np.ndarray, float]:
        return eye[[(3 * moment + 1) % 4]], (1.0 if action == moment % 2 else -1.0)

    brain = calm_brain(seed=5, youth=0 if not wake else 4, need=need)
    action = brain.live(eye[[0]])
    for moment in range(3 if wake else 1):
        x, reward = world(int(action[0]), moment)
        action = brain.live(x, reward=[0.0 if not wake else reward])
    assert (brain.basal_ganglia._pending is not None) == wake  # an action awaits its outcome
    twin = cd.Brain.load(brain.save(tmp_path / "life.npz"))
    assert twin.arousal.to_dict() == brain.arousal.to_dict()
    actions, twins = [], []
    a, b = action, action.copy()
    for moment in range(3, 43):
        x, reward = world(int(a[0]), moment)
        a = brain.live(x, reward=[reward])
        x, reward = world(int(b[0]), moment)
        b = twin.live(x, reward=[reward])
        actions.append(int(a[0]))
        twins.append(int(b[0]))
        assert brain.last_arousal == twin.last_arousal
    assert actions == twins
    assert twin.arousal.to_dict() == brain.arousal.to_dict()
    np.testing.assert_array_equal(twin.brain.efficacy, brain.brain.efficacy)
    np.testing.assert_array_equal(twin.hippocampus.consolidated, brain.hippocampus.consolidated)
    assert {mode for mode in ("routine", "aroused") if brain.arousal.moments[mode]} == {
        "routine",
        "aroused",
    }


def test_a_brain_of_zero_need_continues_and_an_earlier_arousal_is_refused(tmp_path):
    """The need scale of zero, the expert route's value, loads as saved and continues; a brain
    file holding an arousal of an earlier format is refused, naming the release that loads it."""
    brain = calm_brain(seed=5, youth=8, need_scale=0.0)
    eye = np.eye(4)
    brain.live(eye[[0]])
    for moment in range(5):
        brain.live(eye[[moment % 4]], reward=[float(moment % 2)])
    assert brain.basal_ganglia._pending is not None
    path = brain.save(tmp_path / "zero.npz")
    twin = cd.Brain.load(path)
    assert twin.arousal.to_dict() == brain.arousal.to_dict()
    for moment in range(40):
        x, reward = eye[[moment % 4]], [float(moment % 3 == 0) - 0.5]
        np.testing.assert_array_equal(brain.live(x, reward=reward), twin.live(x, reward=reward))
        assert brain.last_arousal == twin.last_arousal
    assert brain.arousal.to_dict() == twin.arousal.to_dict()
    with np.load(path) as archive:
        arrays = flat({key: archive[key] for key in archive.files})
    meta = metadata(arrays)
    meta["arousal"]["format"] = "cadence-arousal/3"
    arrays["generic"] = np.array(json.dumps(meta))
    write(path, arrays)
    with pytest.raises(ValueError, match="cadence 0.94.0 loads it"):
        cd.Brain.load(path)


def test_a_refused_forecast_changes_nothing_and_the_same_call_can_be_retried():
    brain = calm_brain()
    eye = np.eye(4)
    brain.live(eye[[0]])
    brain.live(eye[[1]], reward=[0.0])
    before, mood, lived = snapshot(brain), brain.arousal.to_dict(), brain._lived
    config = brain.learner.config
    brain.learner.config = replace(config, free_steps=0)
    with pytest.raises(RuntimeError, match="did not settle"):
        brain.live(eye[[2]], reward=[-1.0], done=[True])
    assert brain._lived is lived and brain.arousal.to_dict() == mood
    for name, value in snapshot(brain).items():
        np.testing.assert_array_equal(value, before[name])
    brain.learner.config = config
    brain.live(eye[[2]], reward=[-1.0], done=[True])  # the outcome was not lost
    assert brain.last_arousal["mode"] == "aroused" and brain.last_arousal["recorded"]


def test_a_refused_feedback_update_keeps_the_outcome_for_one_retry(monkeypatch):
    brain = calm_brain(seed=4, youth=10)
    eye = np.eye(4)
    action = brain.live(eye[[0]])
    brain.live(eye[[1]], reward=[float(action[0] == 0)])
    assert brain.basal_ganglia._pending is not None  # young: sampled, awaiting its outcome
    before, mood, lived = snapshot(brain), brain.arousal.to_dict(), brain._lived
    learn = brain.basal_ganglia.learn

    def refuse(*args, **kwargs):
        raise RuntimeError("the feedback solve refused")

    monkeypatch.setattr(brain.basal_ganglia, "learn", refuse)
    with pytest.raises(RuntimeError, match="feedback solve refused"):
        brain.live(eye[[2]], reward=[1.0])
    assert brain._lived is lived and brain.arousal.to_dict() == mood
    assert brain.basal_ganglia._pending is not None
    for name, value in snapshot(brain).items():
        np.testing.assert_array_equal(value, before[name])
    monkeypatch.setattr(brain.basal_ganglia, "learn", learn)
    updates = brain.basal_ganglia.updates
    brain.live(eye[[2]], reward=[1.0])  # the same outcome, taken once
    assert brain.last_arousal["learned"] and brain.basal_ganglia.updates == updates + 1


def test_a_stepped_brain_continues_through_live_and_reset_begins_calm():
    brain = calm_brain(seed=2)
    eye = np.eye(4)
    action = brain.step(eye[[0]])
    action = brain.step(eye[[1]], reward=[float(action[0] == 0)])
    updates = brain.basal_ganglia.updates
    brain.live(eye[[2]], reward=[1.0])  # adopts the action step sampled
    assert brain.last_arousal["learned"] and brain.basal_ganglia.updates == updates + 1
    brain.reset()
    assert brain._lived is None and brain.last_arousal is None and brain.arousal.level == 0.0
    assert brain.arousal.age == 1  # a new stream, not a new life
    with pytest.raises(RuntimeError, match="preceding action"):
        brain.live(eye[[0]], reward=[1.0])
    brain.live(eye[[0]])
    brain.act(eye[[1]], greedy=True)  # another operation takes the stream
    with pytest.raises(RuntimeError, match="preceding action"):
        brain.live(eye[[2]], reward=[1.0])


def test_a_slotted_brain_lives_and_owns_only_its_best_guess_in_every_slot():
    brain = cd.Brain.compose(
        4,
        4,
        modules=(16,),
        slots=2,
        seed=0,
        working_memory_amplitude=0.3,
        arousal=cd.ArousalConfig(youth=40),
    )
    eye = np.eye(4)
    action = brain.live(eye[[0]])
    owned = []
    for moment in range(120):
        state = brain.basal_ganglia.state
        motor = state.activation[0, brain.motor_index]
        best = [int(np.argmax(motor[:2])), int(np.argmax(motor[2:]))]
        assert brain._lived[5] == (best == action[0].tolist())
        owned.append(brain._lived[5])
        reward = float(action[0, 0] == moment % 2) - float(action[0, 1] != 0)
        action = brain.live(eye[[moment % 4]], reward=[reward])
        assert action.shape == (1, 2)
    assert not all(owned[:40]) and any(owned[:40])  # a sampling youth explores some slot
    assert brain.arousal.moments["routine"] > 0 and brain.hippocampus.writes > 0


def test_live_runs_on_the_torch_backend_and_fades_device_eligibility(tmp_path):
    pytest.importorskip("torch")
    brain = cd.Brain.compose(
        4,
        2,
        modules=(16,),
        seed=0,
        working_memory_amplitude=0.3,
        backend="torch",
        arousal=cd.ArousalConfig(youth=10),
    )
    eye = np.eye(4)
    agent = brain.basal_ganglia
    action = brain.live(eye[[0]])
    for moment in range(12):
        action = brain.live(eye[[moment % 4]], reward=[0.0])
    assert brain.last_arousal["mode"] == "routine"
    if agent._trace_device is not None:  # the eligibility rests on the device
        kept = [tensor.clone() for tensor in agent._trace_device]
        brain.live(eye[[1]], reward=[0.0])
        decay = agent.config.gamma * agent.config.lam
        for tensor, before in zip(agent._trace_device, kept, strict=True):
            assert bool(((tensor - decay * before).abs() <= 1e-6 * before.abs().max()).all())
        brain.live(eye[[2]], reward=[0.0], done=[True])
        assert not any(bool(tensor.any()) for tensor in agent._trace_device)
    twin = cd.Brain.load(brain.save(tmp_path / "life.npz"), backend="torch")
    for moment in range(20):
        reward = [float(moment % 3 == 0) - 1.0]
        np.testing.assert_array_equal(
            brain.live(eye[[moment % 4]], reward=reward),
            twin.live(eye[[moment % 4]], reward=reward),
        )
    assert twin.arousal.to_dict() == brain.arousal.to_dict()
    assert action.shape == (1,)


def test_brains_without_arousal_keep_their_checkpoint_format(tmp_path):
    plain = cd.Brain.compose(4, 2, modules=(8,), seed=0)
    plain.step(np.eye(4)[[0]])
    with np.load(plain.save(tmp_path / "plain.npz")) as data:
        meta = metadata(data)
        assert meta["format"] == "cadence-brain/1" and "arousal" not in meta
        assert not any(key.startswith("lived/") for key in data.files)
    roused = calm_brain()
    roused.live(np.eye(4)[[0]])
    path = roused.save(tmp_path / "roused.npz")
    with np.load(path) as data:
        meta = metadata(data)
        assert meta["format"] == "cadence-brain/1" and meta["lived"]["sampled"] is False
    loaded = cd.Brain.load(path)
    assert loaded.arousal is not None and loaded._lived is not None
    assert loaded._lived[4] is loaded.basal_ganglia.state
    loaded.act(np.eye(4)[[1]], greedy=True)  # another operation takes the loaded stream too
    with pytest.raises(RuntimeError, match="preceding action"):
        loaded.live(np.eye(4)[[2]], reward=[1.0])
