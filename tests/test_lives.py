"""Independent lives over one topology (cadence#274, #167): each life follows the lone brain of its
seed, on the CPU to the bit and on a torch device within its rounding, and nothing one life learns
reaches another. Two compositions: the default one without the store (the recall memory, the
carried gains, the deep teaching point judged by surprise) and the plain one without the recall
memory."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import cadence as cd

INPUTS, ACTIONS = 12, 5
OPTIONS = {"modules": (16,)}
COMPOSITIONS = {"default": {}, "plain": {"recall": False}}
MAPS = ("head_maps", "position_maps", "anchor_maps", "relative_maps", "slope_maps", "grid_maps")


def lone(seed: int, composition: dict | None = None, **place) -> cd.Brain:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cd.Brain.compose(INPUTS, ACTIONS, seed=seed,
                                **{"episodic": False, **OPTIONS, **(composition or {})}, **place)


def some_lives(count: int, composition: dict | None = None, **place) -> cd.Lives:
    return cd.Lives(INPUTS, ACTIONS, count, **OPTIONS, **(composition or {}), **place)


def a_life(lives: cd.Lives, brains: list[cd.Brain], moments: int, seed: int = 0,
           ends=None, teaches: int = 3) -> int:
    """Step the lives and the lone brains alike: rewards, endings and a teacher every third
    moment; returns how many actions differed. ``ends(t)`` gives the moment's endings in place
    of random ones."""
    rng = np.random.default_rng(seed)
    count, differed = lives.lives, 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for t in range(moments):
            x = np.eye(INPUTS)[rng.integers(INPUTS, size=count)] * rng.random((count, INPUTS))
            reward, done = rng.random(count), rng.random(count) < 0.15
            if ends is not None:
                done = ends(t)
            teacher = rng.integers(ACTIONS, size=count) if t % teaches == 1 else None
            outcome = {} if t == 0 else {"reward": reward, "done": done}
            got = lives.step(x, teacher=teacher, **outcome)
            for i, brain in enumerate(brains):
                one = {} if t == 0 else {"reward": reward[i : i + 1], "done": done[i : i + 1]}
                taught = None if teacher is None else teacher[i : i + 1]
                differed += int(brain.step(x[i : i + 1], teacher=taught, **one)[0] != got[i])
    return differed


def gaps(lives: cd.Lives, brains: list[cd.Brain]) -> dict[str, float]:
    """The largest difference between each life and its lone brain, by part."""
    out: dict[str, float] = {}

    def widen(name: str, ours, theirs) -> None:
        gap = float(np.abs(np.asarray(ours, dtype=float) - np.asarray(theirs, dtype=float)).max())
        out[name] = max(out.get(name, 0.0), gap)

    for i, brain in enumerate(brains):
        held = lives.parameters(i)
        learner, agent = brain.learner, brain.basal_ganglia
        widen("efficacy", held["efficacy"], brain.brain.efficacy)
        widen("bias", held["bias"], brain.brain.bias)
        widen("w_critic", held["w_critic"], agent.w_critic)
        widen("b_critic", held["b_critic"], agent.b_critic)
        widen("velocity", held["velocity"], learner.velocity)
        widen("second_moment", held["second_moment"], learner.second_moment)
        assert held["contrast_updates"] == learner.contrast_updates
        assert held["updates"] == learner.updates
        if brain.recall is not None:
            for name, maps in held["recall"].items():
                for ours, theirs in zip(maps, getattr(brain.recall, name)[0], strict=True):
                    widen(name, ours, theirs)
        if brain.hearing is not None:
            for name, gain in held["carried_gains"].items():
                widen("gain " + name, gain, brain.carried_gains[name])
            widen("carried_second", held["carried_second"], brain.hearing.second)
            assert held["carried_count"] == brain.hearing.count
        store = brain.hippocampus
        if store is not None:
            records = held["store"]
            widen("store", records["consolidated"], store.consolidated)
            assert records["writes"] == store.writes
            if store.gate_noise is not None:
                widen("gate_noise", records["gate_noise"], store.gate_noise)
            if store.fade:
                widen("fade_depth", records["fade_depth"], store.fade_depth)
                widen("fade_recent", records["fade_recent"], store.fade_recent)
        if agent.maturity is not None:
            widen("maturity_use", held["maturity_use"], agent.maturity.use)
    return out


@pytest.mark.parametrize("composition", COMPOSITIONS.values(), ids=COMPOSITIONS)
def test_lives_on_the_cpu_are_lone_brains_to_the_bit(composition):
    lives = cd.Lives(INPUTS, ACTIONS, 3, seeds=[4, 5, 6], **OPTIONS, **composition)
    brains = [lone(s, composition, backend="cpu") for s in (4, 5, 6)]
    assert a_life(lives, brains, 12) == 0
    assert max(gaps(lives, brains).values()) == 0.0


torch = pytest.importorskip("torch")
DEVICES = ["cpu", *(["cuda"] if torch.cuda.is_available() else [])]


@pytest.mark.parametrize("composition", COMPOSITIONS.values(), ids=COMPOSITIONS)
@pytest.mark.parametrize("device", DEVICES)
def test_each_life_on_a_device_follows_its_lone_brain_in_float64(device, composition):
    place = {"backend": "torch", "device": device, "precision": "float64"}
    lives = some_lives(6, composition, **place)
    brains = [lone(s, composition, **place) for s in range(6)]
    assert a_life(lives, brains, 30) == 0
    found = gaps(lives, brains)
    assert max(found.values()) < 1e-9, found
    moved = max(
        float(np.abs(lives.parameters(i)["efficacy"] - lone(i, composition, backend="cpu")
                     .brain.efficacy).max())
        for i in range(6)
    )
    assert moved > 0.05  # the lives learned


@pytest.mark.parametrize(
    ("actions", "options"),
    [(5, {"recall_address_reward": 0.01}), (6, {"slots": 2}), (5, {"modules": (16, 8)}),
     (5, {"carried_gain": 0.0}), (5, {"recall_position": 0}), (5, {"observers": (8,)}),
     (5, {"teaching": "founder"}), (5, {"learning_surprise": False})],
    ids=["address reward", "two slots", "two regions", "no carried gain", "no position term",
         "an observer", "founder teaching", "every row taught"],
)
def test_variants_of_the_composition_follow_their_lone_brains(actions, options):
    place = {"backend": "torch", "device": "cpu", "precision": "float64"}
    composed = {**OPTIONS, **options}
    lives = cd.Lives(INPUTS, actions, 3, **place, **composed)
    rng = np.random.default_rng(2)
    slots = options.get("slots", 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        brains = [cd.Brain.compose(INPUTS, actions, seed=s, episodic=False, **place, **composed)
                  for s in range(3)]
        for t in range(18):
            x = np.eye(INPUTS)[rng.integers(INPUTS, size=3)] * rng.random((3, INPUTS))
            reward, done = rng.random(3), rng.random(3) < 0.15
            teacher = None
            if t % 3 == 1:
                shape = (3, slots) if slots > 1 else (3,)
                teacher = rng.integers(actions // slots, size=shape)
            outcome = {} if t == 0 else {"reward": reward, "done": done}
            got = lives.step(x, teacher=teacher, **outcome)
            for i, brain in enumerate(brains):
                one = {} if t == 0 else {"reward": reward[i : i + 1], "done": done[i : i + 1]}
                taught = None if teacher is None else teacher[i : i + 1]
                np.testing.assert_array_equal(brain.step(x[i : i + 1], teacher=taught, **one)[0],
                                              got[i])
    found = gaps(lives, brains)
    assert max(found.values()) < 1e-9, found


def test_the_default_lives_learn_every_part_each_on_its_own_lessons():
    """The parts the default composition adds are exercised and differ between lives: each life's
    own count of lessons that taught it, its recall maps and its carried gain."""
    place = {"backend": "torch", "device": "cpu", "precision": "float64"}
    lives = some_lives(6, **place)
    brains = [lone(s, **place) for s in range(6)]
    assert a_life(lives, brains, 36) == 0
    held = [lives.parameters(i) for i in range(6)]
    assert len({one["contrast_updates"] for one in held}) > 1  # judged life by life
    for name in ("head_maps", "position_maps", "anchor_maps", "slope_maps"):
        assert max(float(np.abs(one["recall"][name][0]).max()) for one in held) > 1e-4, name
    assert len({round(one["carried_gains"]["trace"], 9) for one in held}) > 1
    assert max(gaps(lives, brains).values()) < 1e-9


def test_streams_of_uneven_length_past_the_position_table_follow_their_lone_brains():
    """One life's stream ends while the others' run past the position table: the slope's rule
    reads each life's own slots, and the grid maps learn."""
    place = {"backend": "torch", "device": "cpu", "precision": "float64"}
    lives = some_lives(4, **place)
    brains = [lone(s, **place) for s in range(4)]

    def ends(t: int) -> np.ndarray:
        done = np.zeros(4, dtype=bool)
        done[1], done[2] = t in (36, 44), t == 20
        return done

    assert a_life(lives, brains, 56, ends=ends) == 0
    found = gaps(lives, brains)
    assert max(found.values()) < 1e-9, found
    assert brains[0].recall.slots > brains[0].recall.position > brains[1].recall.slots
    assert max(
        float(np.abs(lives.parameters(i)["recall"]["grid_maps"][0]).max()) for i in range(4)
    ) > 1e-4


@pytest.mark.parametrize("composition", COMPOSITIONS.values(), ids=COMPOSITIONS)
@pytest.mark.parametrize("device", DEVICES)
def test_each_life_in_float32_follows_its_lone_brain_within_rounding(device, composition):
    """A float32 life follows its lone brain within the spread float32 gives a lone brain's own
    two memory paths: every action the same over these moments, the parameters near."""
    place = {"backend": "torch", "device": device, "precision": "float32"}
    lives = some_lives(4, composition, **place)
    brains = [lone(s, composition, **place) for s in range(4)]
    assert a_life(lives, brains, 16) == 0
    found = gaps(lives, brains)
    assert max(found.values()) < 5e-3, found


@pytest.mark.parametrize("composition", COMPOSITIONS.values(), ids=COMPOSITIONS)
def test_a_solve_that_needs_its_fallback_and_its_rescue_follows_the_lone_brains(composition):
    """A budget of eight sweeps: some lives qualify at once, some at the half step, some on the
    damping ladder. Each life's solve is its own."""
    place = {"backend": "torch", "device": "cpu", "precision": "float64"}
    tight = {**composition, "learning_free_steps": 8}
    lives = some_lives(6, tight, **place)
    brains = [lone(s, tight, **place) for s in range(6)]
    rng = np.random.default_rng(0)
    ways = set()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for t in range(8):
            x = rng.random((6, INPUTS)) * 3
            reward = rng.random(6)
            outcome = {} if t == 0 else {"reward": reward}
            got = lives.step(x, **outcome)
            for i, brain in enumerate(brains):
                one = {} if t == 0 else {"reward": reward[i : i + 1]}
                assert brain.step(x[i : i + 1], **one)[0] == got[i]
                report = brain.last_settlement
                ways.add((report["damping_halvings"], "rescue_steps" in report))
    assert {(0, False), (1, False), (0, True)} <= ways or {(0, False), (1, True)} <= ways, ways
    assert max(gaps(lives, brains).values()) < 1e-9


@pytest.mark.parametrize("composition", COMPOSITIONS.values(), ids=COMPOSITIONS)
def test_greedy_acts_and_lessons_follow_the_lone_brains(composition):
    place = {"backend": "torch", "device": "cpu", "precision": "float64"}
    lives = some_lives(4, composition, **place)
    brains = [lone(s, composition, **place) for s in range(4)]
    rng = np.random.default_rng(3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(8):
            x = np.eye(INPUTS)[rng.integers(INPUTS, size=4)]
            y = rng.integers(ACTIONS, size=4)
            lives.lesson(x, y)
            got = lives.act(x, greedy=True)
            for i, brain in enumerate(brains):
                brain.lesson(x[i : i + 1], y[i : i + 1])
                assert brain.act(x[i : i + 1], greedy=True)[0] == got[i]
    assert max(gaps(lives, brains).values()) < 1e-9


@pytest.mark.parametrize("composition", COMPOSITIONS.values(), ids=COMPOSITIONS)
def test_a_life_learns_nothing_of_another(composition):
    """Two lives of one seed, one rewarded and one not: they part, and each follows its own
    lone brain."""
    place = {"backend": "torch", "device": "cpu", "precision": "float64"}
    lives = cd.Lives(INPUTS, ACTIONS, 2, seeds=[7, 7], **place, **OPTIONS, **composition)
    x = np.eye(INPUTS)[[2, 2]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lives.step(x)
        for _ in range(8):
            lives.step(x, reward=np.array([1.0, 0.0]))
    first, second = lives.parameters(0), lives.parameters(1)
    assert np.abs(first["efficacy"] - second["efficacy"]).max() > 1e-6
    assert abs(float(first["b_critic"]) - float(second["b_critic"])) > 1e-6


@pytest.mark.parametrize("composition", COMPOSITIONS.values(), ids=COMPOSITIONS)
def test_a_life_is_a_lone_brain_that_carries_what_it_learned_and_saves(tmp_path, composition):
    """``life(i)`` is a lone brain on the CPU with everything the life learned: its parameters,
    optimizer, critic, recall maps, carried gains and random state are its lone brain's."""
    place = {"backend": "torch", "device": "cpu", "precision": "float64"}
    lives = some_lives(3, composition, **place)
    brains = [lone(s, composition, **place) for s in range(3)]
    a_life(lives, brains, 12)
    one, beside = lives.life(1), brains[1]
    assert one.placement["backend"] == "cpu"

    def near(ours, theirs, name: str) -> None:
        np.testing.assert_allclose(ours, theirs, rtol=0, atol=1e-9, err_msg=name)

    near(one.brain.efficacy, beside.brain.efficacy, "efficacy")
    near(one.brain.bias, beside.brain.bias, "bias")
    near(one.basal_ganglia.w_critic, beside.basal_ganglia.w_critic, "w_critic")
    near(one.basal_ganglia.b_critic, beside.basal_ganglia.b_critic, "b_critic")
    for name in ("velocity", "velocity_bias", "second_moment", "second_moment_bias"):
        near(getattr(one.learner, name), getattr(beside.learner, name), name)
    assert one.learner.updates == beside.learner.updates
    assert one.learner.contrast_updates == beside.learner.contrast_updates
    assert one.basal_ganglia.rng.bit_generator.state == beside.basal_ganglia.rng.bit_generator.state
    if one.recall is not None:
        for name in MAPS:
            for ours, theirs in zip(getattr(one.recall, name)[0],
                                    getattr(beside.recall, name)[0], strict=True):
                near(ours, theirs, name)
        assert one.carried_gains == pytest.approx(beside.carried_gains, abs=1e-9)
        near(one.hearing.second, beside.hearing.second, "carried second")
        assert one.hearing.count == beside.hearing.count
    x = np.eye(INPUTS)[[3]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        one.step(x)
        one.step(x, reward=np.ones(1), teacher=np.array([2]))  # it lives on as any brain
    loaded = cd.Brain.load(one.save(tmp_path / "life.npz"), backend="cpu")
    np.testing.assert_array_equal(loaded.brain.efficacy, one.brain.efficacy)
    assert loaded.carried_gains == one.carried_gains


def test_a_reset_starts_every_stream_afresh_and_keeps_what_the_lives_learned():
    place = {"backend": "torch", "device": "cpu", "precision": "float64"}
    lives = some_lives(3, **place)
    brains = [lone(s, **place) for s in range(3)]
    a_life(lives, brains, 6)
    learned = lives.parameters(2)["efficacy"]
    lives.reset()
    for brain in brains:
        brain.reset()
    np.testing.assert_array_equal(lives.parameters(2)["efficacy"], learned)
    assert a_life(lives, brains, 9, seed=1) == 0
    assert max(gaps(lives, brains).values()) < 1e-9


def test_the_lives_recall_memory_rests_on_the_device_alone():
    lives = some_lives(2, backend="torch", device="cpu")
    with pytest.raises(RuntimeError, match="rests on their device"):
        lives.brain.recall.keys  # noqa: B018  # a host read of the lives' slots


@pytest.mark.parametrize(
    ("options", "named"),
    [({"episodic": True, "memory_choice": True}, "memory_choice"),
     ({"apical": 0.25, "modules": (16, 8)}, "apical"), ({"recall_heads": 2}, "one head"),
     ({"efference_amplitude": 1.0}, "efference"), ({"learning_rehearsal": 0.5}, "rehearsal"),
     ({"recall_cortex_norm": True}, "cortex at unit norm")],
)
def test_lives_refuse_a_part_their_axis_does_not_cover(options, named):
    with pytest.raises(ValueError, match=named):
        cd.Lives(INPUTS, ACTIONS, 2, backend="torch", device="cpu", **{**OPTIONS, **options})


def test_lives_refuse_a_code_drawn_from_a_seed():
    with pytest.raises(ValueError, match="code of the observation"):
        cd.Lives(32, ACTIONS, 2, backend="torch", device="cpu", code=64, **OPTIONS)


def test_a_moment_of_lives_takes_one_row_per_life():
    lives = some_lives(3, backend="torch", device="cpu")
    with pytest.raises(ValueError, match="one row"):
        lives.step(np.eye(INPUTS)[[1, 2]])
    with pytest.raises(ValueError, match="one seed per life"):
        cd.Lives(INPUTS, ACTIONS, 3, seeds=[1, 2], **OPTIONS)
    with pytest.raises(ValueError, match="vector of inputs"):
        cd.Lives((8, 8), ACTIONS, 2)


# -- continuing lives: ``live`` with each life's own arousal, outcomes and endings (cadence#293)

AROUSED = {"arousal": True, "arousal_youth": 6}  # a short youth: calm and aroused moments mix


def lived(lives: cd.Lives, brains: list[cd.Brain], moments: int, seed: int = 0,
          teaches: int = 5, slots: int = 1, jolts: float = 0.0) -> dict[str, int]:
    """Live the lives and the lone brains alike: sparse rewards, endings, and every ``teaches``
    moments a step with a teacher in place of ``live``. Returns the count of differing actions
    and moods, and of the calm, aroused and heated moments seen."""
    rng = np.random.default_rng(seed)
    count = lives.lives
    seen = {"actions": 0, "moods": 0, "calm": 0, "aroused": 0, "heated": 0, "taught": 0,
            "recorded": 0}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for t in range(moments):
            x = np.eye(INPUTS)[rng.integers(INPUTS, size=count)] * rng.random((count, INPUTS))
            reward = rng.random(count) * (rng.random(count) < 0.5)
            if jolts:  # an occasional outcome far from the usual: it wakes a calm life
                reward = np.where(rng.random(count) < jolts, -20.0 * rng.random(count), reward)
            done = rng.random(count) < 0.15
            labels = lives.actions // slots
            teacher = rng.integers(labels, size=(count, slots) if slots > 1 else count)
            if t and t % teaches == 0:  # a teacher's moment: the pending outcomes, no others
                pending = np.array([b.basal_ganglia._pending is not None for b in brains])
                got = lives.step(x, reward=np.where(pending, reward, 0.0),
                                 done=done & pending, teacher=teacher)
                for i, brain in enumerate(brains):
                    one = ({"reward": reward[i : i + 1], "done": done[i : i + 1]}
                           if pending[i] else {})
                    a = brain.step(x[i : i + 1], teacher=teacher[i : i + 1], **one)
                    seen["actions"] += int(not np.array_equal(a[0], got[i]))
                seen["taught"] += 1
                continue
            takes = np.array([cd.lives._takes_outcome(b) for b in brains])
            outcome = {"reward": np.where(takes, reward, 0.0), "done": done & takes}
            got = lives.live(x, **outcome)
            for i, brain in enumerate(brains):
                one = ({"reward": reward[i : i + 1], "done": done[i : i + 1]}
                       if takes[i] else {})
                a = brain.live(x[i : i + 1], **one)
                seen["actions"] += int(not np.array_equal(a[0], got[i]))
                ours, theirs = lives.last_arousal[i], brain.last_arousal
                seen["moods"] += int(ours["mode"] != theirs["mode"]
                                     or ours["heated_slot"] != theirs["heated_slot"]
                                     or ours["recorded"] != theirs["recorded"]
                                     or abs(ours["level"] - theirs["level"]) > 1e-9)
                seen["calm" if theirs["mode"] == "routine" else "aroused"] += 1
                seen["heated"] += int(theirs["heated_slot"] is not None)
                seen["recorded"] += int(theirs["recorded"])
    return seen


def mood_gaps(lives: cd.Lives, brains: list[cd.Brain]) -> float:
    gap = 0.0
    for ours, brain in zip(lives.arousals, brains, strict=True):
        a, b = ours.to_dict(), brain.arousal.to_dict()
        assert a["age"] == b["age"] and a["outcomes"] == b["outcomes"]
        assert a["rewards"] == b["rewards"] and a["spreads"] == b["spreads"]
        assert a["moments"] == b["moments"]
        gap = max(gap, *(abs(a[k] - b[k]) for k in ("level", "square", "recent", "longrun",
                                                   "usual")))
    return gap


@pytest.mark.parametrize("composition", COMPOSITIONS.values(), ids=COMPOSITIONS)
def test_living_lives_on_the_cpu_are_lone_brains_to_the_bit(composition):
    composition = {**composition, **AROUSED}
    lives = cd.Lives(INPUTS, ACTIONS, 3, seeds=[4, 5, 6], **OPTIONS, **composition)
    brains = [lone(s, composition, backend="cpu") for s in (4, 5, 6)]
    seen = lived(lives, brains, 16, jolts=0.05)
    assert seen["actions"] == seen["moods"] == 0, seen
    assert max(gaps(lives, brains).values()) == 0.0
    assert mood_gaps(lives, brains) == 0.0


@pytest.mark.parametrize(
    ("composition", "actions"),
    [(COMPOSITIONS["default"], ACTIONS), (COMPOSITIONS["plain"], ACTIONS), ({"slots": 2}, 6)],
    ids=["default", "plain", "two slots"],
)
@pytest.mark.parametrize("device", DEVICES)
def test_each_living_life_on_a_device_follows_its_lone_brain_in_float64(
    device, composition, actions
):
    composition = {**composition, **AROUSED}
    place = {"backend": "torch", "device": device, "precision": "float64"}
    lives = cd.Lives(INPUTS, actions, 6, **OPTIONS, **composition, **place)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        brains = [cd.Brain.compose(INPUTS, actions, seed=s, episodic=False, **OPTIONS,
                                   **composition, **place) for s in range(6)]
    seen = lived(lives, brains, 30, slots=composition.get("slots", 1))
    assert seen["actions"] == seen["moods"] == 0, seen
    assert seen["calm"] > 20 and seen["aroused"] > 20 and seen["heated"] > 0, seen
    found = gaps(lives, brains)
    assert max(found.values()) < 1e-9, found
    assert mood_gaps(lives, brains) < 1e-9
    for i, brain in enumerate(brains):
        assert lives.life(i).arousal.to_dict() == lives.arousals[i].to_dict()
        assert lives.life(i).basal_ganglia.updates == brain.basal_ganglia.updates


def test_a_life_without_a_preceding_action_takes_no_outcome():
    lives = some_lives(3, AROUSED, backend="torch", device="cpu", precision="float64")
    x = np.eye(INPUTS)[[1, 2, 3]]
    with pytest.raises(RuntimeError, match="preceding action"):
        lives.live(x, reward=np.array([0.0, 1.0, 0.0]))
    with pytest.raises(RuntimeError, match="preceding action"):
        lives.live(x, done=np.array([False, False, True]))
    assert lives.arousals[0].age == 0
    lives.live(x, reward=np.zeros(3), done=np.zeros(3, bool))  # a zero outcome is none
    assert [a.age for a in lives.arousals] == [1, 1, 1]
    assert lives.last_arousal is not None and not any(r["learned"] for r in lives.last_arousal)


def test_live_needs_each_life_composed_with_arousal():
    lives = some_lives(2, backend="torch", device="cpu")
    with pytest.raises(ValueError, match="arousal"):
        lives.live(np.eye(INPUTS)[[1, 2]])


# -- the store of place records, its fade and maturity (cadence#293)

STORES = {
    "store": {"episodic": True},
    "fading store": {"episodic": True, "memory_fade": 0.05},
    "maturity": {"episodic": True, "maturity": True, "maturity_scale": 0.002},
    "all": {"episodic": True, "memory_fade": 0.05, "maturity": True, "maturity_scale": 0.002},
}


@pytest.mark.parametrize("composition", STORES.values(), ids=STORES)
def test_lives_with_the_store_on_the_cpu_are_lone_brains_to_the_bit(composition):
    composition = {**composition, **AROUSED}
    lives = cd.Lives(INPUTS, ACTIONS, 3, seeds=[4, 5, 6], **OPTIONS, **composition)
    brains = [lone(s, composition, backend="cpu") for s in (4, 5, 6)]
    seen = lived(lives, brains, 16, jolts=0.05)
    assert seen["actions"] == seen["moods"] == 0, seen
    assert max(gaps(lives, brains).values()) == 0.0


@pytest.mark.parametrize("composition", STORES.values(), ids=STORES)
@pytest.mark.parametrize("device", DEVICES)
def test_each_life_with_the_store_follows_its_lone_brain_in_float64(device, composition):
    composition = {**composition, **AROUSED}
    place = {"backend": "torch", "device": device, "precision": "float64"}
    lives = some_lives(6, composition, **place)
    brains = [lone(s, composition, **place) for s in range(6)]
    seen = lived(lives, brains, 30, jolts=0.05)
    assert seen["actions"] == seen["moods"] == 0, seen
    assert seen["calm"] > 20 and seen["aroused"] > 20 and seen["recorded"] > 0, seen
    found = gaps(lives, brains)
    assert max(found.values()) < 1e-9, found
    assert mood_gaps(lives, brains) < 1e-9
    for i, brain in enumerate(brains):
        assert int(lives.parameters(i)["store"]["writes"]) == brain.hippocampus.writes
    assert sum(b.hippocampus.writes for b in brains) > 20  # the records were written
    if composition.get("maturity"):  # matured: the gate engaged, the steps slowed
        assert max(b.basal_ganglia.maturity.factor() for b in brains) < 0.5
    copy = lives.life(2)  # life 2 alone carries its records and its maturity
    np.testing.assert_allclose(copy.hippocampus.consolidated,
                               brains[2].hippocampus.consolidated, atol=1e-9)


@pytest.mark.parametrize("composition", STORES.values(), ids=STORES)
def test_stepping_lives_with_the_store_follow_their_lone_brains(composition):
    place = {"backend": "torch", "device": "cpu", "precision": "float64"}
    lives = some_lives(4, composition, **place)
    brains = [lone(s, composition, **place) for s in range(4)]
    assert a_life(lives, brains, 24) == 0
    found = gaps(lives, brains)
    assert max(found.values()) < 1e-9, found


# -- a life leaving and rejoining (cadence#293)

def moments(rng: np.random.Generator, count: int):
    """A moment's observations, sparse rewards with an occasional jolt, and endings."""
    x = np.eye(INPUTS)[rng.integers(INPUTS, size=count)] * rng.random((count, INPUTS))
    reward = rng.random(count) * (rng.random(count) < 0.5)
    reward = np.where(rng.random(count) < 0.05, -20.0 * rng.random(count), reward)
    return x, reward, rng.random(count) < 0.15


def live_one(brain: cd.Brain, x, reward, done) -> int:
    """``live`` of a lone brain on one row, with its outcome where it has an action out."""
    one = {"reward": reward, "done": done} if cd.lives._takes_outcome(brain) else {}
    return int(brain.live(x, **one)[0])


def live_alike(lives: cd.Lives, brains: list[cd.Brain], x, reward, done) -> int:
    """One moment of ``live`` for the lives and their lone brains; the actions that differed."""
    takes = np.array([cd.lives._takes_outcome(b) for b in brains])
    got = lives.live(x, reward=np.where(takes, reward, 0.0), done=done & takes)
    return sum(
        live_one(b, x[i : i + 1], reward[i : i + 1], done[i : i + 1]) != got[i]
        for i, b in enumerate(brains)
    )


LEAVING = {"default": {}, "store": STORES["all"]}
PLACE = {"backend": "torch", "device": "cpu", "precision": "float64"}


@pytest.mark.parametrize("composition", LEAVING.values(), ids=LEAVING)
def test_a_life_leaves_with_its_stream_and_its_pending_outcome(composition):
    composition = {**composition, **AROUSED}
    lives = some_lives(4, composition, **PLACE)
    brains = [lone(s, composition, **PLACE) for s in range(4)]
    rng = np.random.default_rng(5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(14):
            assert live_alike(lives, brains, *moments(rng, 4)) == 0
        pendings = [lives.life(i, stream=True, **PLACE).basal_ganglia._pending is not None
                    for i in range(4)]
        assert pendings == [b.basal_ganglia._pending is not None for b in brains]
        assert any(pendings) and not all(pendings)  # sampled and calm lives both leave
        left = [lives.life(i, stream=True, **PLACE) for i in range(4)]
        for _ in range(12):  # each copy continues its life as its reference does
            x, reward, done = moments(rng, 4)
            for i in range(4):
                ours = live_one(left[i], x[i : i + 1], reward[i : i + 1], done[i : i + 1])
                assert ours == live_one(brains[i], x[i : i + 1], reward[i : i + 1],
                                        done[i : i + 1])
    for copy, brain in zip(left, brains, strict=True):
        assert copy.arousal.to_dict()["level"] == pytest.approx(
            brain.arousal.to_dict()["level"], abs=1e-9)
        for ours, theirs in ((copy.brain.efficacy, brain.brain.efficacy),
                             (copy.brain.bias, brain.brain.bias),
                             (copy.basal_ganglia.w_critic, brain.basal_ganglia.w_critic)):
            assert float(np.abs(np.asarray(ours) - np.asarray(theirs)).max()) < 1e-9


@pytest.mark.parametrize("composition", LEAVING.values(), ids=LEAVING)
def test_a_life_joins_where_another_finished_and_lives_on_as_alone(composition):
    composition = {**composition, **AROUSED}
    lives = some_lives(4, composition, **PLACE)
    brains = [lone(s, composition, **PLACE) for s in range(4)]
    newcomer = lone(9, composition, **PLACE)
    rng, own = np.random.default_rng(7), np.random.default_rng(8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(10):
            assert live_alike(lives, brains, *moments(rng, 4)) == 0
            x, reward, done = moments(own, 1)
            live_one(newcomer, x, reward, done)
        newcomer.reset()
        lives.join(1, newcomer)  # life 1's place: the newcomer, its stream afresh
        brains[1] = newcomer
        for _ in range(14):
            assert live_alike(lives, brains, *moments(rng, 4)) == 0
    found = gaps(lives, brains)
    assert max(found.values()) < 1e-9, found
    assert mood_gaps(lives, brains) < 1e-9


def test_a_joining_brain_has_the_lives_composition():
    lives = some_lives(2, AROUSED, **PLACE)
    with pytest.raises(ValueError, match="composition"):
        lives.join(0, lone(5, {}, **PLACE))
    with pytest.raises(ValueError, match="one of these"):
        lives.join(2, lone(5, AROUSED, **PLACE))
