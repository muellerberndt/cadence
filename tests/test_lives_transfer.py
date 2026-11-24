"""A life leaves the lives and a brain joins them, on the CPU and on a torch device alike. The
brain ``life(i)`` returns is its own: its stream starts afresh, or with ``stream=True`` stands as
the life's, and nothing it lives through reaches the lives. The lives keep a brain that joins as
a life of their own, its stream afresh, and the brain stays its caller's."""

from __future__ import annotations

import copy
import warnings

import numpy as np
import pytest

import cadence as cd

INPUTS, ACTIONS, LIVES = 12, 5, 3
OPTIONS = {"modules": (16,), "arousal": True, "arousal_youth": 6}
CPU = {"backend": "cpu"}
TORCH = {"backend": "torch", "device": "cpu", "precision": "float64"}


@pytest.fixture(params=[CPU, TORCH], ids=["cpu", "torch"])
def place(request) -> dict:
    """Where the lives rest: lone brains on the CPU, or one brain on the torch CPU device."""
    if request.param is TORCH:
        pytest.importorskip("torch")
    return request.param


def near(place: dict) -> float:
    """How near a lone brain stays to its life: to the bit on the CPU, where the life is a lone
    brain, and within 1e-9 beside the lives of a device."""
    return 0.0 if place is CPU else 1e-9


def some_lives(place: dict) -> cd.Lives:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cd.Lives(INPUTS, ACTIONS, LIVES, **OPTIONS, **place)


def lone(seed: int, place: dict) -> cd.Brain:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cd.Brain.compose(INPUTS, ACTIONS, seed=seed, **OPTIONS, **place)


def moment(seed: int, count: int = LIVES):
    """A moment's observations, rewards and endings, one row per life."""
    rng = np.random.default_rng(seed)
    x = np.eye(INPUTS)[rng.integers(INPUTS, size=count)] * rng.random((count, INPUTS))
    return x, rng.random(count), rng.random(count) < 0.15


def lived(stepped, moments: int, count: int = LIVES) -> np.ndarray:
    """``moments`` steps of rewards and endings, for lives or a lone brain: each then holds a
    sampled action that awaits its outcome. Returns the last observations."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for t in range(moments):
            x, reward, done = moment(t, count)
            stepped.step(x, **({} if t == 0 else {"reward": reward, "done": done}))
    return x


def parts(held: dict) -> dict[str, np.ndarray]:
    """A life's learned state (``Lives.parameters``) as arrays by name."""
    out: dict[str, np.ndarray] = {}

    def put(name: str, value) -> None:
        if isinstance(value, dict):
            for key, inner in value.items():
                put(f"{name}/{key}", inner)
        elif isinstance(value, tuple):
            for k, inner in enumerate(value):
                put(f"{name}/{k}", inner)
        else:
            out[name] = np.asarray(value)

    put("life", held)
    return out


def same(ours: dict, theirs: dict) -> None:
    """Two lives' learned states are the same to the bit."""
    a, b = parts(ours), parts(theirs)
    assert a.keys() == b.keys()
    for name in a:
        np.testing.assert_array_equal(a[name], b[name], err_msg=name)


def holds(brain: cd.Brain, held: dict, atol: float = 0.0) -> None:
    """A lone brain holds a life's learned state as ``Lives.parameters`` gives it."""
    learner, agent, store = brain.learner, brain.basal_ganglia, brain.hippocampus
    theirs = {
        "efficacy": brain.brain.efficacy, "bias": brain.brain.bias,
        "w_critic": agent.w_critic, "b_critic": agent.b_critic,
        "velocity": learner.velocity, "second_moment": learner.second_moment,
    }
    for name, value in theirs.items():
        np.testing.assert_allclose(held[name], value, rtol=0, atol=atol, err_msg=name)
    np.testing.assert_allclose(held["store"]["consolidated"], store.consolidated, rtol=0,
                               atol=atol, err_msg="store")
    assert held["updates"] == learner.updates
    assert held["contrast_updates"] == learner.contrast_updates
    assert held["store"]["writes"] == store.writes


def saved(brain: cd.Brain, path) -> dict[str, np.ndarray]:
    """A brain's checkpoint as its arrays."""
    with np.load(brain.save(path, compressed=False)) as data:
        return {name: data[name].copy() for name in data.files}


def alike(ours: dict[str, np.ndarray], theirs: dict[str, np.ndarray]) -> None:
    assert ours.keys() == theirs.keys()
    for name in ours:
        assert np.array_equal(ours[name], theirs[name]), name


def test_a_life_leaves_as_a_brain_of_its_own_whose_stream_starts_afresh(place):
    lives, twin = some_lives(place), some_lives(place)
    x = lived(lives, 4)
    lived(twin, 4)
    held = lives.parameters(0)
    out = lives.life(0)
    assert out.placement["backend"] == "cpu"
    assert out.basal_ganglia.state is None and not out.pending_feedback
    assert len(out.hippocampus.strength) == 0  # the store's residual is the stream's
    holds(out, held)
    assert lives.life(0, stream=True).pending_feedback  # the life still awaits its outcome
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out.step(x[:1])  # what the brain lives through alone reaches no life
        out.step(x[:1], reward=np.ones(1), teacher=np.array([2]))
        same(lives.parameters(0), held)
        for seed in (11, 12):  # and the lives go on as lives no brain left
            x, reward, done = moment(seed)
            np.testing.assert_array_equal(lives.step(x, reward=reward, done=done),
                                          twin.step(x, reward=reward, done=done))
    for i in range(LIVES):
        same(lives.parameters(i), twin.parameters(i))


def test_a_life_that_leaves_with_its_stream_and_the_life_it_left_go_on_apart(place):
    """The brain carries the life's sampled action and takes that action's outcome as the life
    does: each lives every moment once, and they stay the same."""
    lives = some_lives(place)
    lived(lives, 4)
    out = lives.life(1, stream=True, **place)
    assert out.pending_feedback
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for seed in (11, 12, 13):
            x, reward, done = moment(seed)
            ours = lives.step(x, reward=reward, done=done)
            assert out.step(x[1:2], reward=reward[1:2], done=done[1:2])[0] == ours[1]
    holds(out, lives.parameters(1), near(place))


def test_a_brain_joins_as_a_life_of_the_lives_own_and_stays_its_callers(tmp_path, place):
    """The joined life begins a stream afresh and lives on as the brain would after a reset.
    The brain keeps its own stream, its pending action among it: the lives' moments do not reach
    it, and what it lives through alone does not reach the lives."""
    lives, twin = some_lives(place), some_lives(place)
    lived(lives, 4)
    lived(twin, 4)
    newcomer = lone(9, place)
    own = lived(newcomer, 3, count=1)
    assert newcomer.pending_feedback
    before = saved(newcomer, tmp_path / "before.npz")
    reference = copy.deepcopy(newcomer)
    reference.reset()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lives.join(1, newcomer)
        twin.join(1, copy.deepcopy(newcomer))
        alike(saved(newcomer, tmp_path / "joined.npz"), before)
        assert not lives.life(1, stream=True).pending_feedback
        holds(newcomer, lives.parameters(1))
        for t, seed in enumerate((11, 12, 13)):
            x, reward, done = moment(seed)
            if t == 0:  # the joined life has no action out yet
                reward[1], done[1] = 0.0, False
            got = lives.step(x, reward=reward, done=done)
            np.testing.assert_array_equal(got, twin.step(x, reward=reward, done=done))
            outcome = {} if t == 0 else {"reward": reward[1:2], "done": done[1:2]}
            assert reference.step(x[1:2], **outcome)[0] == got[1]
            if t == 0:
                alike(saved(newcomer, tmp_path / "after.npz"), before)
                newcomer.step(own, reward=np.ones(1))  # its own outcome, alone
    holds(reference, lives.parameters(1), near(place))
    for i in range(LIVES):
        same(lives.parameters(i), twin.parameters(i))


def test_a_life_that_joins_beside_itself_is_a_second_life(place):
    """Life 0 leaves with its stream and joins in life 1's place: two lives of one learned
    state. Life 0 stands as it stood, its action awaiting its outcome, life 1 begins a stream
    afresh, and their own outcomes part them."""
    lives = some_lives(place)
    lived(lives, 4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lives.join(1, lives.life(0, stream=True))
    assert lives.life(0, stream=True).pending_feedback
    assert not lives.life(1, stream=True).pending_feedback
    same(lives.parameters(0), lives.parameters(1))
    if lives.brains is not None:
        assert lives.brains[0] is not lives.brains[1]
    x = moment(11)[0]
    x[1] = x[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lives.step(x, reward=np.array([1.0, 0.0, 0.0]))
        lives.step(x, reward=np.array([1.0, -1.0, 0.0]))
    first, second = lives.parameters(0), lives.parameters(1)
    assert np.abs(first["efficacy"] - second["efficacy"]).max() > 1e-6
    assert first["updates"] != second["updates"]


def test_saved_copies_of_a_life_are_alike_whatever_an_earlier_copy_lived_through(
    tmp_path, place
):
    """Two copies of one life, saved, hold the same arrays, the second taken after the first
    lived on alone; each file loads as a brain that goes on as the life does."""
    lives = some_lives(place)
    x = lived(lives, 4)
    first = lives.life(2, stream=True, **place)
    early = saved(first, tmp_path / "first.npz")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        first.step(x[2:], reward=np.ones(1))
        second = lives.life(2, stream=True, **place)
        alike(saved(second, tmp_path / "second.npz"), early)
        loaded = [cd.Brain.load(tmp_path / name, **place) for name in ("first.npz", "second.npz")]
        for seed in (11, 12):
            x, reward, done = moment(seed)
            got = lives.step(x, reward=reward, done=done)
            for brain in loaded:
                assert brain.step(x[2:], reward=reward[2:], done=done[2:])[0] == got[2]
    for brain in loaded:
        holds(brain, lives.parameters(2), near(place))


def test_a_copy_of_a_life_on_the_cpu_shares_its_anatomy_and_nothing_it_can_change():
    """The copy is the life's lone brain copied deep: the immutable connectome is shared, every
    array is its own, and each random generator is its own at the life's state."""
    lives = some_lives(CPU)
    lived(lives, 4)
    life = lives.brains[0]
    carried = lives.life(0, stream=True)
    np.testing.assert_array_equal(carried.hippocampus.strength, life.hippocampus.strength)
    assert len(carried.hippocampus.strength) == 1
    for out in (carried, lives.life(0)):
        assert out is not life
        assert out.connectome is life.connectome
        assert not np.shares_memory(out.brain.efficacy, life.brain.efficacy)
        assert not np.shares_memory(out.hippocampus.consolidated, life.hippocampus.consolidated)
        for ours, theirs in ((out.rng, life.rng), (out.basal_ganglia.rng, life.basal_ganglia.rng)):
            assert ours is not theirs
            assert ours.bit_generator.state == theirs.bit_generator.state
        assert out.arousal is not life.arousal
        assert out.arousal.to_dict() == life.arousal.to_dict()
    joining = lone(9, CPU)
    lives.join(1, joining)
    assert lives.brains[1] is not joining
    assert lives.brains[1].connectome is joining.connectome
    assert not np.shares_memory(lives.brains[1].brain.efficacy, joining.brain.efficacy)


def test_a_life_of_the_cpu_lives_leaves_for_the_place_named():
    """The lives rest on the CPU and the brain on the torch device named: it holds the life's
    learned state, with ``stream=True`` the action that awaits its outcome, and takes that
    outcome there while the life stays as it was."""
    pytest.importorskip("torch")
    lives = some_lives(CPU)
    x = lived(lives, 4)
    held = lives.parameters(0)
    for stream in (False, True):
        out = lives.life(0, stream=stream, **TORCH)
        placed = out.placement
        assert (placed["backend"], placed["device"], placed["precision"]) == (
            "torch", "cpu", "float64")
        assert out.pending_feedback is stream
        holds(out, held, 1e-12)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out.step(x[:1], reward=np.ones(1))
    assert out.learner.updates == held["updates"] + 1
    same(lives.parameters(0), held)
