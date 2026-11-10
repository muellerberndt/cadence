"""``Brain.value`` (cadence#298): the critic's value of an observation for a truncated episode's
bootstrap, read without changing anything."""

from __future__ import annotations

import copy
import warnings

import numpy as np
import pytest

import cadence as cd

INPUTS, ACTIONS = 12, 5


def composed(seed: int = 0, **options) -> cd.Brain:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cd.Brain.compose(INPUTS, ACTIONS, seed=seed, modules=(16,), **options)


def world(seed: int, streams: int = 1):
    rng = np.random.default_rng(seed)
    while True:
        x = np.eye(INPUTS)[rng.integers(INPUTS, size=streams)] * rng.random((streams, INPUTS))
        yield x, rng.random(streams), rng.random(streams) < 0.1


def lived(brain: cd.Brain, moments: int, seed: int = 0, streams: int = 1) -> np.ndarray:
    """``moments`` steps of rewards and endings; the last observation."""
    feed = world(seed, streams)
    x = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for t in range(moments):
            x, reward, done = next(feed)
            brain.step(x, **({} if t == 0 else {"reward": reward, "done": done}))
    return x


def arrays(value, seen=None, path="brain"):
    """Every array reachable from ``value``'s attributes, by path, the caches of spare arrays
    (``_rooms``) and of a graph's flat weights (``_flat``) left out."""
    seen = set() if seen is None else seen
    if id(value) in seen or path.endswith(("_rooms", "_flat")):
        return {}
    seen.add(id(value))
    if isinstance(value, np.ndarray):
        return {path: value}
    if isinstance(value, np.random.Generator):
        return {path: np.frombuffer(repr(value.bit_generator.state).encode(), dtype=np.uint8)}
    out = {}
    if isinstance(value, dict):
        items = value.items()
    elif isinstance(value, (list, tuple)):
        items = enumerate(value)
    elif hasattr(value, "__dict__") and type(value).__module__.startswith("cadence"):
        items = vars(value).items()
    else:
        return {}
    for key, item in items:
        out.update(arrays(item, seen, f"{path}.{key}"))
    return out


COMPOSITIONS = {
    "default": {}, "without the store": {"episodic": False}, "plain": {"recall": False},
    "arousal": {"arousal": True}, "learned code": {"learned_code": 8},
}


@pytest.mark.parametrize("options", COMPOSITIONS.values(), ids=COMPOSITIONS)
def test_a_read_changes_no_array_of_the_brain(options):
    brain = composed(**options)
    x = lived(brain, 9)
    before = {k: v.copy() for k, v in arrays(brain).items()}
    value = brain.value(x)
    assert value.shape == (1,) and np.isfinite(value).all()
    after = arrays(brain)
    assert before.keys() == after.keys()
    changed = [k for k in before if not np.array_equal(before[k], after[k])]
    assert not changed, changed


@pytest.mark.parametrize("options", COMPOSITIONS.values(), ids=COMPOSITIONS)
def test_a_brain_that_reads_lives_on_to_the_bit_as_one_that_does_not(options):
    reading, plain = composed(**options), composed(**options)
    x = lived(reading, 9)
    lived(plain, 9)
    reading.value(x)
    feed_a, feed_b = world(3), world(3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(8):
            (xa, ra, da), (xb, rb, db) = next(feed_a), next(feed_b)
            assert reading.step(xa, reward=ra, done=da)[0] == plain.step(xb, reward=rb, done=db)[0]
    np.testing.assert_array_equal(reading.brain.efficacy, plain.brain.efficacy)
    np.testing.assert_array_equal(reading.basal_ganglia.w_critic, plain.basal_ganglia.w_critic)


@pytest.mark.parametrize("options", COMPOSITIONS.values(), ids=COMPOSITIONS)
def test_the_read_is_the_critics_value_of_the_drive_the_brain_reads(options):
    brain = composed(**options)
    x = lived(brain, 9)
    other = copy.deepcopy(brain)
    agent = other.basal_ganglia
    other._prepare(x)
    drive = other.stimulus(x)
    warm = agent.state if other.learned_code is None else other._warm(drive, agent.state)
    np.testing.assert_array_equal(brain.value(x), agent.value(agent.learner.free(drive, warm=warm)))


@pytest.mark.parametrize(
    "options", [{"episodic": False}, {"recall": False, "episodic": False}], ids=["recall", "plain"]
)
def test_a_truncated_row_bootstrapped_by_its_value_learns_as_one_that_goes_on(options):
    """Without the store (whose record of the outcome ``learn`` writes before it reads the next
    state), a row marked ``done`` with ``bootstrap=brain.value(last)`` takes the dopamine of a
    row that goes on, to the bit."""
    on, cut = composed(**options), composed(**options)
    lived(on, 9, streams=3)
    x = lived(cut, 9, streams=3)
    reward = np.array([0.3, 0.0, 1.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        on.learn(reward, np.zeros(3, bool), x)
        cut.learn(reward, np.ones(3, bool), x, bootstrap=cut.value(x))
    np.testing.assert_array_equal(on.brain.efficacy, cut.brain.efficacy)
    np.testing.assert_array_equal(on.brain.bias, cut.brain.bias)
    np.testing.assert_array_equal(on.basal_ganglia.w_critic, cut.basal_ganglia.w_critic)
    assert on.basal_ganglia.b_critic == cut.basal_ganglia.b_critic


def test_a_brain_that_has_not_acted_reads_from_rest():
    brain = composed()
    x = np.eye(INPUTS)[[1, 4]]
    before = {k: v.copy() for k, v in arrays(brain).items()}
    value = brain.value(x)
    assert value.shape == (2,)
    assert all(np.array_equal(before[k], v) for k, v in arrays(brain).items() if k in before)
    agent = brain.basal_ganglia
    np.testing.assert_array_equal(value, agent.value(agent.learner.free(brain.stimulus(x))))


@pytest.mark.parametrize(
    "observations",
    [np.full((1, INPUTS), np.nan), np.zeros((1, INPUTS + 1)), np.zeros((2, INPUTS)),
     np.zeros(0)],
    ids=["not finite", "too wide", "another batch", "empty"],
)
def test_malformed_observations_are_refused_before_any_change(observations):
    brain = composed()
    lived(brain, 6)
    before = {k: v.copy() for k, v in arrays(brain).items()}
    with pytest.raises(ValueError):
        brain.value(observations)
    after = arrays(brain)
    assert all(np.array_equal(before[k], after[k]) for k in before)


torch = pytest.importorskip("torch")


def test_a_read_on_a_device_changes_nothing_the_brain_goes_on_with():
    place = {"backend": "torch", "device": "cpu", "precision": "float64"}
    reading, plain = composed(**place), composed(**place)
    x = lived(reading, 9)
    lived(plain, 9)
    value = reading.value(x)
    assert np.isfinite(value).all()
    feed_a, feed_b = world(3), world(3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(8):
            (xa, ra, da), (xb, rb, db) = next(feed_a), next(feed_b)
            assert reading.step(xa, reward=ra, done=da)[0] == plain.step(xb, reward=rb, done=db)[0]
    np.testing.assert_array_equal(reading.brain.efficacy, plain.brain.efficacy)


@pytest.mark.parametrize("backend", ["cpu", "torch"])
def test_each_lifes_value_is_its_lone_brains(backend):
    place = {"backend": "torch", "device": "cpu", "precision": "float64"} if backend == "torch" \
        else {"backend": "cpu"}
    lives = cd.Lives(INPUTS, ACTIONS, 3, modules=(16,), **place)
    brains = [composed(s, episodic=False, **place) for s in range(3)]
    feed = world(4, 3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for t in range(8):
            x, reward, done = next(feed)
            outcome = {} if t == 0 else {"reward": reward, "done": done}
            lives.step(x, **outcome)
            for i, b in enumerate(brains):
                b.step(x[i : i + 1], **({} if t == 0 else
                                       {"reward": reward[i : i + 1], "done": done[i : i + 1]}))
    ours = lives.values(x)
    theirs = np.concatenate([b.value(x[i : i + 1]) for i, b in enumerate(brains)])
    if backend == "cpu":
        np.testing.assert_array_equal(ours, theirs)
    else:
        np.testing.assert_allclose(ours, theirs, atol=1e-9)
