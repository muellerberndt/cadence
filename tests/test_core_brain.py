"""A composed brain's moment in the native core (``cadence.core.parts.brain``): its state crosses
and comes back the same bytes on both sides, every ``live`` call of a life gives in the core the
action the Python gives and leaves the state it leaves (within the rounding of the block products'
order, which natively NumPy sums in its BLAS's order), and a brain held in the core lives the
same life call after call. The brain is composed as Wild composes an animal's: senses by name, a
sparse code, the deep lesson, the recall memory with two heads, bound records with a voice,
surprise with its focus, a map of things and a maturity."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pytest

import cadence as cd
from cadence.core import Core, Refused, _default_library, pack
from cadence.core.parts import part
from cadence.space.map import Map

BRAIN = part("brain")
SENSES = {"eyes": 16, "heard": 8, "food": {"what": 4, "where": 4}, "body": 4}


def brain(seed: int = 3) -> Any:
    held = cd.Brain.compose(
        senses=SENSES, actions=4, code=32, modules=(10, 10), apical=0.25, teaching="deep",
        recall_capacity=6, recall_heads=2, bound=5, voice="heard", maturity=True,
        surprise={"hold": 0.8, "reach": 3.0}, arousal={"youth": 6, "threshold": 0.2},
        backend="cpu", seed=seed)
    held.space = Map((0.0, 12.0, 0.0, 12.0), (4, 4), units=2, values={"hue": 2}, reach=3.0,
                     step=1.8, stride=6.0, margin=0.3, physics={"noise": 0.3}, surprise=True)
    held.space.moment(seen=[(0, [3.0, 4.0], {"hue": np.array([1.0, 0.0])}), (1, [9.0, 8.0])])
    return held


def senses(rng: np.random.Generator) -> dict[str, Any]:
    out: dict[str, Any] = {"eyes": rng.random((1, 16)), "body": rng.random((1, 4))}
    if rng.random() < 0.5:
        out["heard"] = rng.random((1, 8))
    if rng.random() < 0.7:
        out["food"] = {"what": rng.random((1, 4)), "where": rng.random((1, 4))}
    return out


def moments(count: int, seed: int) -> list[tuple[dict[str, Any], float | None]]:
    rng = np.random.default_rng(seed)
    return [(senses(rng), None if t == 0 else float(rng.random() - 0.4)) for t in range(count)]


def lived(held: Any, given: dict[str, Any], reward: float | None) -> Any:
    feedback = {"reward": [reward]} if reward is not None and held.pending_feedback else {}
    return held.live(given, **feedback)


def arguments(held: Any, given: dict[str, Any], reward: float | None) -> dict[str, Any]:
    feedback = {"reward": [reward]} if reward is not None and held.pending_feedback else {}
    _, _, after = BRAIN.calling(held, "live", (given,), feedback)
    return {**BRAIN._live(given, **feedback), **after()}


def same(ours: dict[str, Any], theirs: dict[str, Any], atol: float = 1e-9) -> None:
    assert set(ours) == set(theirs), set(ours) ^ set(theirs)
    for key, value in theirs.items():
        x, y = np.asarray(ours[key]), np.asarray(value)
        assert x.shape == y.shape, key
        if x.dtype.kind == "f":
            np.testing.assert_allclose(x, y, rtol=0, atol=atol, err_msg=key)
        else:
            assert np.array_equal(x, y.astype(x.dtype)), key


def test_state_round_trips_in_python():
    held = brain()
    for t, (given, reward) in enumerate(moments(12, seed=1)):
        lived(held, given, reward)
        if t in (0, 1, 11):
            state = BRAIN.state_of(held)
            fresh = copy.deepcopy(held)
            BRAIN.restore(fresh, {k: np.array(v) for k, v in state.items()})
            assert pack(BRAIN.state_of(fresh)) == pack(state)


def test_the_senses_cross_by_name():
    state = BRAIN.state_of(brain())
    spans = {k[len("senses."):]: tuple(v) for k, v in state.items() if k.startswith("senses.")}
    held = brain()
    assert spans["food"] == (held.senses["food"]["what"].start, held.senses["food"]["where"].stop)
    assert spans["food/where"] == (held.senses["food"]["where"].start,
                                   held.senses["food"]["where"].stop)
    assert spans["voice"] == (held.senses["voice"].start, held.senses["voice"].stop)


core_built = pytest.mark.skipif(not _default_library().exists(),
                                reason="the core library is not built")


@core_built
def test_state_round_trips_in_the_core():
    core = Core()
    held = brain()
    for t, (given, reward) in enumerate(moments(8, seed=2)):
        lived(held, given, reward)
        if t in (0, 7):
            state = BRAIN.state_of(held)
            after, out = core.call("brain", "state", state, {})
            assert not out and pack(after) == pack(state)


@core_built
def test_a_life_s_moments_in_the_core():
    core = Core()
    held = brain(seed=5)
    learned = aroused = 0
    for given, reward in moments(24, seed=4):
        before = BRAIN.state_of(held)
        args = arguments(held, given, reward)
        action = lived(held, given, reward)
        after, out = core.call("brain", "live", before, args)
        assert np.array_equal(out["action"], action)
        same(after, BRAIN.state_of(held))
        learned += bool(held.last_arousal["learned"])
        aroused += held.last_arousal["mode"] == "aroused"
    assert learned and aroused  # sampled outcomes were learned, routine ones measured


@core_built
def test_a_brain_held_in_the_core_lives_the_same_life():
    core = Core()
    held = brain(seed=7)
    handle = core.brain(BRAIN.state_of(held))
    try:
        for given, reward in moments(30, seed=6):
            args = arguments(held, given, reward)
            action = lived(held, given, reward)
            out = core.brain_call(handle, "live", args)
            assert np.array_equal(out["action"], action)
        same(core.brain_state(handle), BRAIN.state_of(held))
    finally:
        core.brain_drop(handle)


@core_built
def test_an_act_in_the_core():
    core = Core()
    held = brain(seed=9)
    for given, reward in moments(6, seed=8):
        lived(held, given, reward)
    for greedy in (False, True):
        given = moments(1, seed=10 + greedy)[0][0]
        before = BRAIN.state_of(held)
        _, _, after = BRAIN.calling(held, "act", (given,), {"greedy": greedy})
        action = held.act(given, greedy=greedy)
        theirs, out = core.call("brain", "act", before,
                                {**BRAIN._act(given, greedy=greedy), **after()})
        assert np.array_equal(out["action"], action)
        same(theirs, BRAIN.state_of(held))


def shown(t: int) -> tuple[np.ndarray, list[float]]:
    """A watching lesson's targets (one action, one-hot) and weight; every third a weight of
    one, the lesson without weights."""
    targets = np.zeros((1, 4))
    targets[0, (3 * t) % 4] = 1.0
    return targets, [1.0 if t % 3 == 0 else 0.25 + 0.05 * (t % 5)]


def lesson_arguments(held: Any, given: dict[str, Any], targets: Any, weights: Any) -> dict[str, Any]:
    _, _, after = BRAIN.calling(held, "lesson", (given,), {})
    return {**BRAIN._lesson(given, targets=targets, weights=weights), **after()}


@core_built
def test_lessons_in_the_core():
    core = Core()
    held = brain(seed=11)
    taken = refused = 0
    for t, (given, reward) in enumerate(moments(40, seed=12)):
        targets, weights = shown(t)
        before = BRAIN.state_of(held)
        args = lesson_arguments(held, given, targets, weights)
        try:
            held.lesson(given, targets=targets, weights=weights)
        except RuntimeError:  # a sampled action's outcome is pending
            refused += 1
            with pytest.raises(Refused):
                core.call("brain", "lesson", before, args)
        else:
            taken += 1
            theirs, out = core.call("brain", "lesson", before, args)
            assert not out
            same(theirs, BRAIN.state_of(held))
        lived(held, given, reward)
    assert taken and refused


@core_built
def test_a_brain_held_in_the_core_takes_the_same_lessons():
    core = Core()
    held = brain(seed=13)
    handle = core.brain(BRAIN.state_of(held))
    try:
        for t, (given, reward) in enumerate(moments(30, seed=14)):
            targets, weights = shown(t)
            args = lesson_arguments(held, given, targets, weights)
            try:
                held.lesson(given, targets=targets, weights=weights)
            except RuntimeError:
                with pytest.raises(Refused):
                    core.brain_call(handle, "lesson", args)
            else:
                core.brain_call(handle, "lesson", args)
            args = arguments(held, given, reward)
            action = lived(held, given, reward)
            assert np.array_equal(core.brain_call(handle, "live", args)["action"], action)
        same(core.brain_state(handle), BRAIN.state_of(held))
    finally:
        core.brain_drop(handle)


@core_built
def test_lessons_through_live_in_the_core():
    core = Core()
    held = brain(seed=19)
    handle = core.brain(BRAIN.state_of(held))
    try:
        for t, (given, reward) in enumerate(moments(30, seed=20)):
            targets, weights = shown(t)
            lesson = (targets, weights) if t % 4 else None  # most moments teach
            feedback = {"reward": [reward]} if reward is not None and held.pending_feedback else {}
            kwargs = {**feedback, **({} if lesson is None else {"lesson": lesson})}
            _, _, after = BRAIN.calling(held, "live", (given,), kwargs)
            args = {**BRAIN._live(given, **kwargs), **after()}
            before = BRAIN.state_of(held)
            updates = held.learner.updates
            action = held.live(given, **kwargs)
            if lesson is not None:  # taken whatever was pending before the moment
                assert held.learner.updates > updates
            theirs, out = core.call("brain", "live", before, args)
            assert np.array_equal(out["action"], action)
            same(theirs, BRAIN.state_of(held))
            assert np.array_equal(core.brain_call(handle, "live", args)["action"], action)
        same(core.brain_state(handle), BRAIN.state_of(held))
    finally:
        core.brain_drop(handle)


@core_built
def test_the_core_refuses_the_lessons_the_python_refuses():
    core = Core()
    held = brain(seed=15)
    for given, reward in moments(12, seed=16):  # past its youth, routine
        lived(held, given, reward)
    given = moments(1, seed=17)[0][0]
    state = BRAIN.state_of(held)
    for targets, weights in ((np.full((1, 4), 0.3), None), (np.eye(4)[:1], [-1.0]),
                             (np.eye(4)[:1], [0.5, 0.5]), (np.eye(5)[:1], None)):
        with pytest.raises(ValueError):
            copy.deepcopy(held).lesson(given, targets=targets, weights=weights)
        with pytest.raises(Refused):
            core.call("brain", "lesson", state, lesson_arguments(held, given, targets, weights))
        with pytest.raises(ValueError):  # and given to live
            copy.deepcopy(held).live(given, lesson=(targets, weights))
        args = {**BRAIN._live(given, lesson=(targets, weights)),
                **BRAIN.calling(held, "live", (given,), {})[2]()}
        with pytest.raises(Refused):
            core.call("brain", "live", state, args)


@core_built
def test_the_core_refuses_what_the_python_refuses():
    core = Core()
    held = brain()
    given, _ = moments(1, seed=0)[0]
    state = BRAIN.state_of(held)
    row = BRAIN.whole_row(held, given)
    with pytest.raises(RuntimeError):  # feedback before any action
        copy.deepcopy(held).live(given, reward=[1.0])
    with pytest.raises(Refused):
        core.call("brain", "live", state, {"observations": row, "mapping": np.bool_(True),
                                           "reward": np.ones(1)})
    bad = row.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        copy.deepcopy(held).live({**given, "eyes": bad[:, :16]})
    with pytest.raises(Refused):
        core.call("brain", "live", state, {"observations": bad, "mapping": np.bool_(True)})
    lived(held, given, None)
    state = BRAIN.state_of(held)
    with pytest.raises(ValueError):  # an outcome for another action
        copy.deepcopy(held).live(given, reward=[0.5], decision_id=held.decision_id + 1)
    with pytest.raises(Refused):
        core.call("brain", "live", state, {"observations": row, "mapping": np.bool_(True),
                                           "reward": np.full(1, 0.5),
                                           "decision_id": np.int64(held.decision_id + 1)})
