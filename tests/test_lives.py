"""Independent lives over one topology (cadence#274, #167): each life follows the lone brain of its
seed, on the CPU to the bit and on a torch device within its rounding, and nothing one life learns
reaches another."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import cadence as cd

INPUTS, ACTIONS = 12, 5
OPTIONS = {"modules": (16,)}


def lone(seed: int, **place) -> cd.Brain:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cd.Brain.compose(INPUTS, ACTIONS, seed=seed, recall=False, episodic=False,
                                **OPTIONS, **place)


def a_life(lives: cd.Lives, brains: list[cd.Brain], moments: int, seed: int = 0) -> int:
    """Step the lives and the lone brains alike: rewards, endings and a teacher now and then;
    returns how many actions differed."""
    rng = np.random.default_rng(seed)
    count, differed = lives.lives, 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for t in range(moments):
            x = np.eye(INPUTS)[rng.integers(INPUTS, size=count)] * rng.random((count, INPUTS))
            reward, done = rng.random(count), rng.random(count) < 0.15
            teacher = rng.integers(ACTIONS, size=count) if t % 5 == 3 else None
            outcome = {} if t == 0 else {"reward": reward, "done": done}
            got = lives.step(x, teacher=teacher, **outcome)
            for i, brain in enumerate(brains):
                one = {} if t == 0 else {"reward": reward[i : i + 1], "done": done[i : i + 1]}
                taught = None if teacher is None else teacher[i : i + 1]
                differed += int(brain.step(x[i : i + 1], teacher=taught, **one)[0] != got[i])
    return differed


def gaps(lives: cd.Lives, brains: list[cd.Brain]) -> dict[str, float]:
    out = {"efficacy": 0.0, "bias": 0.0, "w_critic": 0.0, "b_critic": 0.0}
    for i, brain in enumerate(brains):
        held = lives.parameters(i)
        out["efficacy"] = max(out["efficacy"], float(np.abs(held["efficacy"] - brain.brain.efficacy).max()))
        out["bias"] = max(out["bias"], float(np.abs(held["bias"] - brain.brain.bias).max()))
        agent = brain.basal_ganglia
        out["w_critic"] = max(out["w_critic"], float(np.abs(held["w_critic"] - agent.w_critic).max()))
        out["b_critic"] = max(out["b_critic"], abs(float(held["b_critic"]) - agent.b_critic))
    return out


def test_lives_on_the_cpu_are_lone_brains_to_the_bit():
    lives = cd.Lives(INPUTS, ACTIONS, 3, seeds=[4, 5, 6], **OPTIONS)
    brains = [lone(s, backend="cpu") for s in (4, 5, 6)]
    assert a_life(lives, brains, 12) == 0
    for i, brain in enumerate(brains):
        held = lives.parameters(i)
        np.testing.assert_array_equal(held["efficacy"], brain.brain.efficacy)
        np.testing.assert_array_equal(held["w_critic"], brain.basal_ganglia.w_critic)


torch = pytest.importorskip("torch")
DEVICES = ["cpu", *(["cuda"] if torch.cuda.is_available() else [])]


@pytest.mark.parametrize("device", DEVICES)
def test_each_life_on_a_device_follows_its_lone_brain_in_float64(device):
    place = {"backend": "torch", "device": device, "precision": "float64"}
    lives = cd.Lives(INPUTS, ACTIONS, 6, **place, **OPTIONS)
    brains = [lone(s, **place) for s in range(6)]
    assert a_life(lives, brains, 24) == 0
    found = gaps(lives, brains)
    assert max(found.values()) < 1e-10, found
    moved = max(
        float(np.abs(lives.parameters(i)["efficacy"] - lone(i, backend="cpu").brain.efficacy).max())
        for i in range(6)
    )
    assert moved > 0.05  # the lives learned


@pytest.mark.parametrize("device", DEVICES)
def test_each_life_in_float32_follows_its_lone_brain_within_rounding(device):
    place = {"backend": "torch", "device": device, "precision": "float32"}
    lives = cd.Lives(INPUTS, ACTIONS, 4, **place, **OPTIONS)
    brains = [lone(s, **place) for s in range(4)]
    assert a_life(lives, brains, 16) == 0
    found = gaps(lives, brains)
    assert max(found.values()) < 1e-3, found


def test_greedy_acts_and_lessons_follow_the_lone_brains():
    place = {"backend": "torch", "device": "cpu", "precision": "float64"}
    lives = cd.Lives(INPUTS, ACTIONS, 4, **place, **OPTIONS)
    brains = [lone(s, **place) for s in range(4)]
    rng = np.random.default_rng(3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(6):
            x = np.eye(INPUTS)[rng.integers(INPUTS, size=4)]
            y = rng.integers(ACTIONS, size=4)
            lives.lesson(x, y)
            got = lives.act(x, greedy=True)
            for i, brain in enumerate(brains):
                brain.lesson(x[i : i + 1], y[i : i + 1])
                assert brain.act(x[i : i + 1], greedy=True)[0] == got[i]
    assert gaps(lives, brains)["efficacy"] < 1e-10


def test_a_life_learns_nothing_of_another():
    """Two lives of one seed, one rewarded and one not: they part, and each follows its own
    lone brain."""
    place = {"backend": "torch", "device": "cpu", "precision": "float64"}
    lives = cd.Lives(INPUTS, ACTIONS, 2, seeds=[7, 7], **place, **OPTIONS)
    x = np.eye(INPUTS)[[2, 2]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lives.step(x)
        for _ in range(8):
            lives.step(x, reward=np.array([1.0, 0.0]))
    first, second = lives.parameters(0), lives.parameters(1)
    assert np.abs(first["efficacy"] - second["efficacy"]).max() > 1e-6
    assert abs(float(first["b_critic"]) - float(second["b_critic"])) > 1e-6


def test_a_life_is_a_lone_brain_that_saves(tmp_path):
    place = {"backend": "torch", "device": "cpu", "precision": "float64"}
    lives = cd.Lives(INPUTS, ACTIONS, 3, **place, **OPTIONS)
    brains = [lone(s, **place) for s in range(3)]
    a_life(lives, brains, 8)
    one = lives.life(1)
    held = lives.parameters(1)
    np.testing.assert_allclose(one.brain.efficacy, held["efficacy"])
    np.testing.assert_allclose(one.basal_ganglia.w_critic, held["w_critic"])
    loaded = cd.Brain.load(one.save(tmp_path / "life.npz"), backend="cpu")
    np.testing.assert_array_equal(loaded.brain.efficacy, one.brain.efficacy)
    x = np.eye(INPUTS)[[3]]
    assert loaded.act(x, greedy=True)[0] == one.act(x, greedy=True)[0]


@pytest.mark.parametrize(
    ("options", "named"),
    [({"recall": True}, "recall=False"), ({"episodic": True}, "episodic=False"),
     ({"arousal": True}, "no arousal"), ({"apical": 0.25, "modules": (16, 8)}, "apical")],
)
def test_lives_refuse_a_part_their_axis_does_not_cover(options, named):
    with pytest.raises(ValueError, match=named):
        cd.Lives(INPUTS, ACTIONS, 2, backend="torch", device="cpu", **{**OPTIONS, **options})


def test_a_moment_of_lives_takes_one_row_per_life():
    lives = cd.Lives(INPUTS, ACTIONS, 3, backend="torch", device="cpu", **OPTIONS)
    with pytest.raises(ValueError, match="one row"):
        lives.step(np.eye(INPUTS)[[1, 2]])
    with pytest.raises(ValueError, match="one seed per life"):
        cd.Lives(INPUTS, ACTIONS, 3, seeds=[1, 2], **OPTIONS)


def test_a_reset_starts_every_stream_afresh_and_keeps_what_the_lives_learned():
    place = {"backend": "torch", "device": "cpu", "precision": "float64"}
    lives = cd.Lives(INPUTS, ACTIONS, 3, **place, **OPTIONS)
    brains = [lone(s, **place) for s in range(3)]
    a_life(lives, brains, 6)
    learned = lives.parameters(2)["efficacy"]
    lives.reset()
    for brain in brains:
        brain.reset()
    np.testing.assert_array_equal(lives.parameters(2)["efficacy"], learned)
    assert a_life(lives, brains, 6, seed=1) == 0
    assert gaps(lives, brains)["efficacy"] < 1e-10
