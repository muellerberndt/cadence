"""The episodic hippocampus of the composed brain (0.83.0): records vote in the choice."""

from __future__ import annotations

import json

import numpy as np
import pytest

import cadence as cd


def episodic(*args, **options) -> cd.Brain:
    """``Brain.compose`` with the episodic memory, unless ``memory_choice`` is named."""
    return cd.Brain.compose(*args, **{"memory_choice": True, **options})


def one_hot(index: int, width: int, rows: int = 1) -> np.ndarray:
    x = np.zeros((rows, width))
    x[:, index] = 1.0
    return x


def place(brain: cd.Brain, x: np.ndarray, action: int) -> float:
    """The place record of an action: the hippocampus keyed by the observation."""
    return float(brain.hippocampus.recall(x)[0, action])


def vote(brain: cd.Brain, x: np.ndarray, key: np.ndarray | None = None) -> np.ndarray:
    """The records' vote on the observations, for the stream's present context."""
    if key is None and brain.context_memory is not None:
        key = brain._episode_key(x)
    return brain._vote(x, key)


def consolidated(brain: cd.Brain, x: np.ndarray, action: int) -> float:
    """The shared, consolidated place record: a read in another batch size than the stream's."""
    return float(brain.hippocampus.recall(np.repeat(x, 3, axis=0))[0, action])


def test_the_episodic_memory_is_an_opt_in_preset_of_the_composed_brain():
    assert cd.Brain.compose(3, 2, seed=1).describe()["episodic"] is None
    brain = episodic(3, 2, seed=1)
    for memory in (brain.hippocampus, brain.context_memory):
        # the stream's record moves halfway at once and fades; the shared record consolidates
        assert memory.rate == 0.5 and memory.decay == 0.9 and memory.consolidation == 0.05
    described = brain.describe()["episodic"]
    names = ("choice", "context", "replay", "demonstrations")
    assert {key: described[key] for key in names} == {
        "choice": True,
        "context": 0.5,
        "replay": 8,
        "demonstrations": 1.0,
    }
    np.testing.assert_array_equal(brain.hippocampus.pre, brain.sensory_index)
    np.testing.assert_array_equal(
        brain.context_memory.pre, np.concatenate([brain.sensory_index, brain.motor_index])
    )
    for memory in (brain.hippocampus, brain.context_memory):
        assert memory.separator.expansion == 1024 and memory.separator.winners == 24
    assert episodic(3, 2, seed=1, memory_context=0.0).context_memory is None
    released = cd.Brain.compose(3, 2, seed=1)
    assert released.describe()["episodic"] is None and released.context_memory is None
    np.testing.assert_array_equal(released.hippocampus.pre, released.sensory_index)
    assert released.hippocampus.separator is None and released.hippocampus.rate == 1.0
    for field in ("pre", "post", "sign"):
        np.testing.assert_array_equal(
            getattr(brain.connectome, field), getattr(released.connectome, field)
        )
    assert cd.Brain.compose(3, 2, seed=1, episodic=False).hippocampus is None


def test_the_episodic_settings_are_validated_construction_settings():
    with pytest.raises(ValueError, match="memory_context"):
        episodic(3, 2, memory_context=1.0)
    with pytest.raises(ValueError, match="memory_replay"):
        episodic(3, 2, memory_replay=-1)
    with pytest.raises(ValueError, match="memory_replay"):
        episodic(3, 2, memory_replay=True)
    with pytest.raises(ValueError, match="memory_choice must be boolean"):
        episodic(3, 2, memory_choice=1)
    with pytest.raises(ValueError, match="need memory_choice"):
        cd.Brain.compose(3, 2, memory_context=0.5)
    with pytest.raises(ValueError, match="need memory_choice"):
        cd.Brain.compose(3, 2, memory_expansion=1024)
    connectome = episodic(3, 2, seed=2).connectome
    with pytest.raises(ValueError, match="needs the episodic memory"):
        cd.Brain(connectome, episodic=False, memory_choice=True)
    with pytest.raises(ValueError, match="construction setting"):
        episodic(3, 2).retune(memory_context=0.2)
    brain = episodic(3, 2, seed=2)
    brain.retune(memory_rate=0.3, consolidation=0.1)
    assert brain.hippocampus.rate == brain.context_memory.rate == 0.3
    assert brain.hippocampus.consolidation == brain.context_memory.consolidation == 0.1


def test_a_record_votes_in_greedy_and_sampled_choices():
    brain = episodic(3, 2, seed=3)
    x = one_hot(0, 3)
    for _ in range(3):
        brain.reset()
        brain.step(x, teacher=[1])
    brain.reset()
    votes = vote(brain, x)
    assert votes[0, 1] > votes[0, 0]
    assert int(brain.act(x, greedy=True)[0]) == 1
    state = brain.basal_ganglia.state
    plain = brain.basal_ganglia.probabilities(state)
    voted = brain.basal_ganglia.probabilities(state, bias=votes)
    assert voted[0, 1] > plain[0, 1]


def test_the_vote_is_the_mean_of_the_place_and_the_context_records():
    brain = episodic(3, 2, seed=3)
    x = one_hot(0, 3)
    brain.step(x, teacher=[1])
    key = brain._episode_key(x)
    expected = 0.5 * (brain.hippocampus.recall(x) + brain.context_memory.recall(key))
    np.testing.assert_allclose(brain._vote(x, key), expected)


def test_a_demonstration_outlives_a_zero_outcome_in_the_middle_of_an_attempt():
    brain = episodic(3, 3, seed=4, memory_context=0.0)
    start, middle = one_hot(0, 3), one_hot(1, 3)
    brain.step(start, teacher=[2])
    shown = place(brain, start, 2)
    assert shown > 0
    brain._moment = (start.copy(), np.array([2]))  # the shown action was the one executed
    brain.learn(np.zeros(1), np.zeros(1, bool), middle)
    assert place(brain, start, 2) == pytest.approx(shown)
    brain.step(start)  # decide again in the same situation
    brain._moment = (start.copy(), np.array([2]))
    brain.learn(np.zeros(1), np.ones(1, bool), middle)  # the attempt ended with nothing
    assert place(brain, start, 2) < shown


def test_only_known_outcomes_write_and_a_calm_moment_writes_nothing():
    brain = episodic(2, 2, seed=5)
    x = np.array([[1.0, 0.0], [0.0, 1.0]])
    brain.act(x)
    writes = brain.hippocampus.writes, brain.context_memory.writes
    brain.learn(np.zeros(2), np.zeros(2, bool), x[::-1])  # nothing paid, nothing ended
    assert (brain.hippocampus.writes, brain.context_memory.writes) == writes
    brain.act(x)
    brain.learn(np.array([0.0, -1.0]), np.zeros(2, bool), x)  # one stream punished
    assert (brain.hippocampus.writes, brain.context_memory.writes) == (writes[0] + 1, writes[1] + 1)
    brain.act(x)
    brain.learn(np.zeros(2), np.array([False, True]), x)  # one attempt ended with nothing
    assert brain.hippocampus.writes == writes[0] + 2
    calm = episodic(2, 2, seed=5, arousal={"youth": 0})
    calm.live(one_hot(0, 2))
    held = calm.hippocampus.writes
    calm.live(one_hot(1, 2), reward=[0.0])  # as forecast: routine, nothing learned
    assert calm.last_arousal["mode"] == "routine" and calm.hippocampus.writes == held


def context(brain: cd.Brain, key: np.ndarray, action: int) -> float:
    """The context record of an action under a decision key, in reward units."""
    return float(brain._held(None, key[None, :], np.array([action]), brain.context_memory)[0])


def test_an_attempt_that_ends_in_reward_is_replayed_whole_into_both_records():
    brain = episodic(4, 3, seed=6, actor_gamma=0.9)
    plan = [(0, 1), (0, 0), (1, 2), (2, 2)]  # a stall at situation 0, then onwards
    for situation, action in plan:
        x = one_hot(situation, 4)
        brain.act(x)
        entry = brain._paths[0][-1]
        brain._paths[0][-1] = (entry[0], np.array(action), x[0], 0.0, True, 0.5)  # came to nothing
        brain._moment = (x, np.array([action]))
    assert [int(np.argmax(entry[2])) for entry in brain._paths[0]] == [0, 1, 2]
    assert [int(entry[1]) for entry in brain._paths[0]] == [0, 2, 2]
    stalled = place(brain, one_hot(0, 4), 1)
    writes = brain.hippocampus.writes, brain.context_memory.writes
    brain.learn(np.ones(1), np.ones(1, bool), one_hot(3, 4))  # paid, and the attempt ended
    # the paid decision and the two on its path, in both kinds of record
    assert (brain.hippocampus.writes, brain.context_memory.writes) == (
        writes[0] + 3,
        writes[1] + 3,
    )
    assert place(brain, one_hot(0, 4), 1) == pytest.approx(stalled)  # nothing wrote action 1
    # discounted per step of the path: one step back holds more than two
    assert consolidated(brain, one_hot(1, 4), 2) > consolidated(brain, one_hot(0, 4), 0) > 0
    assert brain._paths[0] == []


def test_a_supplied_salience_marks_an_outcome_known_and_consolidates_it_faster():
    def written(salience: list[float] | None) -> tuple[int, float]:
        brain = episodic(3, 2, seed=16, memory_context=0.0)
        x = one_hot(0, 3)
        brain.step(x)
        action = int(brain._moment[1][0])
        brain.step(one_hot(1, 3), reward=[0.5], salience=salience)
        return brain.hippocampus.writes, consolidated(brain, x, action)

    plain, sized = written(None), written([4.0])
    assert plain[0] == sized[0] == 1
    assert sized[1] > plain[1] > 0  # the supplied salience consolidates faster
    brain = episodic(3, 2, seed=16, memory_context=0.0)
    brain.step(one_hot(0, 3))
    brain.step(one_hot(1, 3), reward=[0.0], salience=[1.0])  # nothing paid, but marked
    assert brain.hippocampus.writes == 1


def continuing(seed: int, expected: dict[tuple[int, int], float]):
    """A brain and a ``decide`` that makes a given decision in a continuing life, the decision
    as much the brain's best guess as ``expected`` says (1.0 unless named)."""
    brain = episodic(4, 3, seed=seed, actor_gamma=0.9)
    keys = {}

    def decide(situation: int, action: int, reward: float, following: int) -> None:
        x = one_hot(situation, 4)
        brain.act(x)
        entry = brain._paths[0][-1]
        guess = expected.get((situation, action), 1.0)
        brain._paths[0][-1] = (entry[0], np.array(action), entry[2], 0.0, False, guess)
        keys[situation, action] = entry[0]
        brain._moment = (x, np.array([action]))
        brain.learn(np.array([reward]), np.zeros(1, bool), one_hot(following, 4))

    return brain, keys, decide


def test_a_reward_in_a_continuing_life_credits_the_context_back_to_the_last_known_outcome():
    brain, keys, decide = continuing(6, {})
    plan = [(0, 1), (0, 0), (1, 2), (2, 2)]  # a stall at situation 0, then onwards
    for index, (situation, action) in enumerate(plan[:-1]):
        decide(situation, action, 0.0, plan[index + 1][0])
    assert [int(np.argmax(entry[2])) for entry in brain._paths[0]] == [0, 1]
    assert [int(entry[1]) for entry in brain._paths[0]] == [0, 2]
    stalled = context(brain, keys[0, 1], 1)
    writes = brain.hippocampus.writes, brain.context_memory.writes
    decide(2, 2, 1.0, 3)  # paid; nothing ended
    # the paid decision in both kinds of record; the two best guesses on its path in their
    # context records
    assert (brain.hippocampus.writes, brain.context_memory.writes) == (
        writes[0] + 1,
        writes[1] + 3,
    )
    assert context(brain, keys[0, 1], 1) == pytest.approx(stalled)  # the stall is not replayed
    # discounted per step of the path: one step back holds more than two
    assert context(brain, keys[1, 2], 2) > context(brain, keys[0, 0], 0) > 0
    assert place(brain, one_hot(1, 4), 2) == 0.0  # a best guess gains no place record
    writes = brain.context_memory.writes
    decide(3, 1, 0.0, 1)
    decide(1, 0, 1.0, 0)  # paid again: the replay stops at the earlier reward
    assert brain.context_memory.writes == writes + 2
    assert len(brain._paths[0]) == 5  # nothing ended: the path goes on


def test_an_exploration_that_led_to_a_surprising_reward_also_gains_its_place_record():
    brain, keys, decide = continuing(6, {(1, 2): 0.25})  # explored: far below the best guess
    for situation, action, following in ((0, 0, 1), (1, 2, 2)):
        decide(situation, action, 0.0, following)
    decide(2, 2, 1.0, 3)
    explored = place(brain, one_hot(1, 4), 2)
    assert explored > 0  # the exploration, in place too
    assert explored < context(brain, keys[1, 2], 2)  # in proportion to how unexpected it was
    assert place(brain, one_hot(0, 4), 0) == 0.0  # the best guess before it: context only


def test_replay_credits_the_surprise_of_a_reward_and_an_expected_one_replays_nothing():
    def replayed(surprise: float) -> float:
        brain = episodic(3, 2, seed=7, actor_gamma=0.9)
        brain.act(one_hot(0, 3))
        entry = brain._paths[0][-1]
        brain._paths[0][-1] = (entry[0], np.array(0), entry[2], 0.0, True, 1.0)  # came to nothing
        brain.act(one_hot(1, 3))  # the paid decision, in a continuing life
        brain._replay(np.ones(1), np.array([surprise]), np.zeros(1, bool), np.ones(1, bool))
        assert place(brain, one_hot(0, 3), 0) == 0.0  # a best guess: context only
        return context(brain, entry[0], 0)

    credit = replayed(1.0)  # nothing expected: the whole reward
    assert 0 < credit < 0.9  # toward the discounted reward
    assert replayed(0.5) == pytest.approx(0.5 * credit, rel=0.05)
    assert replayed(0.0) == 0.0  # an expected reward replays nothing
    places = episodic(3, 2, seed=7, memory_context=0.0)
    assert places.context_memory is None  # without context records nothing is replayed


def test_the_temporal_context_recalls_a_cue_that_is_gone():
    brain = episodic(3, 2, seed=8)
    cues = {0: one_hot(0, 3), 1: one_hot(1, 3)}
    go = one_hot(2, 3)
    for cue, shown in cues.items():  # one demonstration of each cue's answer
        brain.reset()
        brain.step(shown, teacher=[cue])
        brain.step(go, reward=[0.0], teacher=[cue])
    for cue, shown in cues.items():
        brain.reset()
        brain.act(shown, greedy=True)
        assert int(brain.act(go, greedy=True)[0]) == cue
    # the place records alone cannot tell the two moments at go apart
    places = episodic(3, 2, seed=8, memory_context=0.0)
    assert places.context_memory is None and places._context is None


def test_what_the_stream_did_enters_its_context():
    brain = episodic(2, 3, seed=9)
    x = one_hot(0, 2)
    action = int(brain.act(x)[0])
    context = brain._context[0]
    assert context[2 + action] == pytest.approx(0.5)
    assert context[0] == pytest.approx(0.5)
    brain.learn(np.zeros(1), np.ones(1, bool), x)
    assert not brain._context.any() and brain._paths == [[]]


def test_a_saved_episodic_brain_continues_exactly(tmp_path):
    brain = episodic(3, 2, seed=10, arousal=True)
    for t in range(4):
        brain.step(one_hot(t % 3, 3), reward=None if t == 0 else [0.0])
    path = brain.save(tmp_path / "brain.npz")
    copy = cd.Brain.load(path)
    assert copy.describe()["episodic"] == brain.describe()["episodic"]
    np.testing.assert_array_equal(copy._context, brain._context)
    np.testing.assert_array_equal(copy._decision, brain._decision)
    assert len(copy._paths[0]) == len(brain._paths[0])
    for t in range(6):
        reward, done = [float(t == 5)], [t == 5]
        np.testing.assert_array_equal(
            brain.step(one_hot(t % 2, 3), reward=reward, done=done),
            copy.step(one_hot(t % 2, 3), reward=reward, done=done),
        )
    for name in ("hippocampus", "context_memory"):
        np.testing.assert_allclose(
            getattr(copy, name).consolidated, getattr(brain, name).consolidated
        )
    meta = json.loads(str(np.load(path)["generic"]))
    assert meta["format"] == "cadence-generic/6"
    places = episodic(3, 2, seed=10, memory_context=0.0)
    places.step(one_hot(0, 3))
    resumed = cd.Brain.load(places.save(tmp_path / "places.npz"))
    assert resumed.context_memory is None and resumed.memory_choice
    released = cd.Brain.compose(3, 2, seed=10)
    released.act(one_hot(0, 3))
    meta = json.loads(str(np.load(released.save(tmp_path / "released.npz"))["generic"]))
    assert meta["format"] == "cadence-generic/2" and "episodic" not in meta


def test_a_calm_brain_records_the_outcome_that_woke_it():
    brain = episodic(2, 2, seed=11, arousal={"youth": 0})
    x = one_hot(0, 2)
    brain.live(x)
    for _ in range(5):  # outcomes as forecast: routine
        brain.live(x, reward=[0.0])
    assert brain.last_arousal["mode"] == "routine"
    action, key = int(brain._lived[1][0]), brain._decision.copy()
    before = vote(brain, x, key)[0, action]
    brain.live(x, reward=[5.0])  # far beyond the forecast: it wakes the calm brain
    assert brain.last_arousal["recorded"]
    assert vote(brain, x, key)[0, action] > before


def test_waiting_drifts_the_context():
    brain = episodic(3, 2, seed=12, arousal=True)
    brain.live(one_hot(0, 3))
    before = brain._context.copy()
    brain.wait(one_hot(1, 3))
    np.testing.assert_allclose(brain._context, 0.5 * before + 0.5 * np.r_[one_hot(1, 3)[0], 0, 0])


def test_slots_take_the_records_vote_per_slot():
    brain = episodic(2, 4, slots=2, seed=13)
    x = one_hot(0, 2)
    for _ in range(3):
        brain.reset()
        brain.step(x, teacher=[[1, 0]])
    brain.reset()
    np.testing.assert_array_equal(brain.act(x, greedy=True)[0], [1, 0])


def test_a_choosing_memory_is_not_read_into_the_drive(monkeypatch):
    brain = episodic(3, 2, seed=14)
    brain.step(one_hot(0, 3), teacher=[1])

    def refused(*args, **kwargs):
        raise AssertionError("the records were read into the drive")

    for memory in (brain.hippocampus, brain.context_memory):
        monkeypatch.setattr(memory, "stimulate", refused)
    brain.stimulus(one_hot(0, 3))
    brain.imagine([one_hot(0, 3), one_hot(1, 3)])
    brain.act(one_hot(0, 3))


@pytest.mark.parametrize(
    "tamper", ["no_context_store", "wide_place_key", "long_path", "no_guess", "odd_guess"]
)
def test_a_corrupt_episodic_checkpoint_is_refused(tmp_path, tamper):
    brain = episodic(3, 2, seed=15)
    for t in range(3):
        brain.step(one_hot(t, 3), reward=None if t == 0 else [0.0])
    path = brain.save(tmp_path / "brain.npz")
    with np.load(path) as saved:
        data = {name: saved[name].copy() for name in saved.files}
    meta = json.loads(str(data["generic"]))
    if tamper == "no_context_store":
        meta["episodic"]["context_memory"] = None
    elif tamper == "wide_place_key":
        meta["hippocampus"]["pre"] = 5
    elif tamper == "no_guess":
        del data["episode/path/guess"]
    elif tamper == "odd_guess":
        data["episode/path/guess"] = data["episode/path/guess"] + 2.0
    else:
        data["episode/path/lengths"] = data["episode/path/lengths"] + 20
    data["generic"] = np.array(json.dumps(meta))
    corrupt = tmp_path / "corrupt.npz"
    np.savez(corrupt, **data)
    with pytest.raises(ValueError):
        cd.Brain.load(corrupt)
