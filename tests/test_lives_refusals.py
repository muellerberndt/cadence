"""What the lives on the CPU refuse. A call's rows, labels, outcomes and their owners are checked
for every life before any life moves: a refused call leaves every life's checkpoint as it was, and
the corrected call behaves as a first call. A life that refuses a call it was validly given stops
the lives in row order, and its exception says which lives have moved."""

from __future__ import annotations

import warnings
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import cadence as cd

INPUTS, ACTIONS, LIVES = 12, 5, 3
OPTIONS = {"modules": (16,), "arousal": True, "arousal_youth": 6}
ONES, NO_END = np.ones(LIVES), np.zeros(LIVES, bool)


def some_lives(**place) -> cd.Lives:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cd.Lives(INPUTS, ACTIONS, LIVES, **OPTIONS, **place)


def rows(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.eye(INPUTS)[rng.integers(INPUTS, size=LIVES)] * rng.random((LIVES, INPUTS))


def spoiled(x: np.ndarray, row: int = 1, value: float = np.nan) -> np.ndarray:
    """``x`` with one value of one life's row not finite."""
    out = x.copy()
    out[row, 3] = value
    return out


def prepared(lives, how: str | None) -> None:
    """The lives before the call: fresh, or after a ``step``, an ``act`` or three of ``live``."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if how == "stepped":
            lives.step(rows(7))
        elif how == "acted":
            lives.act(rows(7))
        elif how == "lived":
            for seed in (7, 8, 9):
                lives.live(rows(seed))


def saved(lives: cd.Lives, folder) -> list[dict[str, np.ndarray]]:
    """Each life's checkpoint as its arrays: its parameters, optimizer, critic, random states,
    stream, pending action, memories and arousal."""
    out = []
    for i, brain in enumerate(lives.brains):
        with np.load(brain.save(folder / f"life{i}.npz", compressed=False)) as data:
            out.append({name: data[name].copy() for name in data.files})
    return out


def alike(ours: list[dict[str, np.ndarray]], theirs: list[dict[str, np.ndarray]]) -> None:
    for i, (a, b) in enumerate(zip(ours, theirs, strict=True)):
        assert a.keys() == b.keys(), i
        for name in a:
            assert np.array_equal(a[name], b[name]), (i, name)


# name: (the lives before the call, the refused call, the call corrected, the refusal)
REFUSED = {
    "act, a row not finite": (
        None, lambda p, x: p.act(spoiled(x)), lambda p, x: p.act(x), ValueError),
    "act, the last row infinite": (
        None, lambda p, x: p.act(spoiled(x, LIVES - 1, np.inf)), lambda p, x: p.act(x),
        ValueError),
    "greedy act, a row not finite": (
        "stepped", lambda p, x: p.act(spoiled(x), greedy=True),
        lambda p, x: p.act(x, greedy=True), ValueError),
    "step, a row not finite": (
        None, lambda p, x: p.step(spoiled(x)), lambda p, x: p.step(x), ValueError),
    "step, a label outside the choices": (
        None, lambda p, x: p.step(x, teacher=[0, ACTIONS, 1]),
        lambda p, x: p.step(x, teacher=[0, 2, 1]), ValueError),
    "step, a negative label": (
        None, lambda p, x: p.step(x, teacher=[0, 1, -1]),
        lambda p, x: p.step(x, teacher=[0, 1, 2]), ValueError),
    "step, a label too many": (
        None, lambda p, x: p.step(x, teacher=[0, 1, 2, 3]),
        lambda p, x: p.step(x, teacher=[0, 1, 2]), ValueError),
    "step, a label too few": (
        None, lambda p, x: p.step(x, teacher=[0, 1]),
        lambda p, x: p.step(x, teacher=[0, 1, 2]), ValueError),
    "step, one label for all": (
        None, lambda p, x: p.step(x, teacher=1), lambda p, x: p.step(x, teacher=[1, 1, 1]),
        ValueError),
    "step with outcomes, a row not finite": (
        "stepped", lambda p, x: p.step(spoiled(x), reward=ONES, done=NO_END),
        lambda p, x: p.step(x, reward=ONES, done=NO_END), ValueError),
    "step with outcomes, a label outside the choices": (
        "stepped", lambda p, x: p.step(x, reward=ONES, teacher=[0, 1, ACTIONS]),
        lambda p, x: p.step(x, reward=ONES, teacher=[0, 1, 2]), ValueError),
    "step with outcomes, a label too many": (
        "stepped", lambda p, x: p.step(x, reward=ONES, teacher=[0, 1, 2, 3]),
        lambda p, x: p.step(x, reward=ONES, teacher=[0, 1, 2]), ValueError),
    "lesson, a row not finite": (
        None, lambda p, x: p.lesson(spoiled(x), [0, 1, 2]), lambda p, x: p.lesson(x, [0, 1, 2]),
        ValueError),
    "lesson, a label outside the choices": (
        None, lambda p, x: p.lesson(x, [0, ACTIONS, 1]), lambda p, x: p.lesson(x, [0, 2, 1]),
        ValueError),
    "lesson, a label too many": (
        None, lambda p, x: p.lesson(x, [0, 1, 2, 3]), lambda p, x: p.lesson(x, [0, 1, 2]),
        ValueError),
    "lesson, a label too few": (
        None, lambda p, x: p.lesson(x, [0, 1]), lambda p, x: p.lesson(x, [0, 1, 2]),
        ValueError),
    "lesson, one label for all": (
        None, lambda p, x: p.lesson(x, 1), lambda p, x: p.lesson(x, [1, 1, 1]), ValueError),
    "lesson, no labels": (
        None, lambda p, x: p.lesson(x, None), lambda p, x: p.lesson(x, [0, 1, 2]), ValueError),
    "learn, a row not finite": (
        "acted", lambda p, x: p.learn(ONES, NO_END, spoiled(x)),
        lambda p, x: p.learn(ONES, NO_END, x), ValueError),
    "learn, no reward and no ending": (
        "acted", lambda p, x: p.learn(None, None, x), lambda p, x: p.learn(ONES, NO_END, x),
        ValueError),
    "live, a row not finite": (
        None, lambda p, x: p.live(spoiled(x)), lambda p, x: p.live(x), ValueError),
    "live with outcomes, a row not finite": (
        "lived", lambda p, x: p.live(spoiled(x), reward=np.zeros(LIVES), done=NO_END),
        lambda p, x: p.live(x, reward=np.zeros(LIVES), done=NO_END), ValueError),
}

# the calls the lives refuse for every life at their boundary on either backend: with the calls
# above, what the lives on the CPU, the lives on a torch device and a lone brain refuse alike
AT_THE_BOUNDARY = {
    "step, a reward not finite": (
        "stepped", lambda p, x: p.step(x, reward=np.array([1.0, np.nan, 0.0])), ValueError),
    "step, a reward too many": (
        "stepped", lambda p, x: p.step(x, reward=np.ones(LIVES + 1)), ValueError),
    "step, endings that are no booleans": (
        "stepped", lambda p, x: p.step(x, done=np.zeros(LIVES, int)), ValueError),
    "step, a reward for an action no life issued": (
        None, lambda p, x: p.step(x, reward=ONES), RuntimeError),
    "learn, an outcome of an action no life issued": (
        None, lambda p, x: p.learn(ONES, NO_END, x), RuntimeError),
    "learn, a reward not finite": (
        "acted", lambda p, x: p.learn(np.array([1.0, np.inf, 0.0]), NO_END, x), ValueError),
    "learn, an ending too many": (
        "acted", lambda p, x: p.learn(ONES, np.zeros(LIVES + 1, bool), x), ValueError),
    "lesson, while sampled actions await their outcomes": (
        "acted", lambda p, x: p.lesson(x, [0, 1, 2]), RuntimeError),
    "values, a row not finite": (
        "stepped", lambda p, x: p.values(spoiled(x)), ValueError),
    "values, a row too few": (
        "stepped", lambda p, x: p.values(x[:-1]), ValueError),
}


@pytest.mark.parametrize("case", REFUSED.values(), ids=REFUSED)
def test_a_refused_call_leaves_every_life_as_it_was_and_its_correction_is_a_first_call(
    tmp_path, case
):
    how, refused, corrected, kind = case
    lives, twin = some_lives(), some_lives()
    prepared(lives, how)
    prepared(twin, how)
    x = rows(1)
    before = saved(lives, tmp_path)
    with pytest.raises(kind):
        refused(lives, x)
    alike(saved(lives, tmp_path), before)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # the corrected call, and a moment after it, beside lives that were never refused
        ours, theirs = corrected(lives, x), corrected(twin, x)
        assert (ours is None and theirs is None) or np.array_equal(ours, theirs)
        np.testing.assert_array_equal(lives.step(rows(2)), twin.step(rows(2)))
    alike(saved(lives, tmp_path), saved(twin, tmp_path))


def test_a_lesson_is_refused_for_every_life_while_one_life_awaits_an_outcome(tmp_path):
    """Life 1 alone holds a sampled action: the lesson teaches no life, the lives before it
    among them."""
    lives = some_lives()
    x = rows(1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lives.brains[1].act(x[1:2])
    before = saved(lives, tmp_path)
    with pytest.raises(RuntimeError, match="awaits its outcome"):
        lives.lesson(x, [0, 1, 2])
    alike(saved(lives, tmp_path), before)
    assert [b.learner.updates for b in lives.brains] == [0, 0, 0]


def test_a_life_whose_brain_holds_other_streams_is_refused_before_any_life_moves(tmp_path):
    lives = some_lives()
    x = rows(1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lives.brains[1].act(np.vstack([x[1:2], x[1:2]]))  # two streams in life 1's brain
    before = saved(lives, tmp_path)
    for call in (lambda: lives.act(x), lambda: lives.step(x), lambda: lives.values(x),
                 lambda: lives.live(x), lambda: lives.learn(ONES, NO_END, x)):
        with pytest.raises(ValueError, match="life 1's brain holds 2"):
            call()
    alike(saved(lives, tmp_path), before)
    assert lives.brains[0].basal_ganglia.state is None


@pytest.mark.parametrize("outcomes", [False, True], ids=["act", "step with outcomes"])
def test_a_life_that_refuses_its_settle_stops_the_lives_and_its_exception_says_so(
    tmp_path, outcomes
):
    """Life 1's settle has a budget of no sweeps, so it refuses as its lone brain does. Life 0
    has completed its part and the exception holds what it returned; life 2 has not begun. The
    lives finished one by one from there are the lives that were never refused; with outcomes
    life 1 has learned its own before its act refused, so it acts and takes no reward again."""
    lives, twin = some_lives(), some_lives()
    prepared(lives, "stepped")
    prepared(twin, "stepped")
    x = rows(1)
    given = {"reward": ONES, "done": NO_END} if outcomes else {}
    learner = lives.brains[1].learner
    usual = learner.config
    before = saved(lives, tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = twin.step(x, **given) if outcomes else twin.act(x)
        learner.config = replace(usual, free_steps=0)
        with pytest.raises(RuntimeError, match="did not settle") as refusal:
            lives.step(x, **given) if outcomes else lives.act(x)
        learner.config = usual
        error = refusal.value
        assert error.life == 1
        assert any("life 1 refused this call" in note for note in error.__notes__)
        assert len(error.completed) == 1
        np.testing.assert_array_equal(error.completed[0], expected[:1])
        after = saved(lives, tmp_path)
        alike(after[2:], before[2:])
        if outcomes:
            assert lives.brains[1].basal_ganglia.updates == 1
            assert not lives.brains[1].pending_feedback
            lives.brains[1].act(x[1:2])
            rest = lives.brains[2].step(x[2:], reward=ONES[2:], done=NO_END[2:])
        else:
            alike(after[1:2], before[1:2])
            np.testing.assert_array_equal(lives.brains[1].act(x[1:2]), expected[1:2])
            rest = lives.brains[2].act(x[2:])
        np.testing.assert_array_equal(rest, expected[2:])
        ours, theirs = saved(lives, tmp_path), saved(twin, tmp_path)
        whole = slice(None) if not outcomes else slice(0, None, 2)  # life 1 learned at no sweeps
        alike(ours[whole], theirs[whole])
        assert all(b.pending_feedback for b in lives.brains)
        lives.step(rows(2), reward=ONES)  # the lives move together again


torch = pytest.importorskip("torch")
PLACE = {"backend": "torch", "device": "cpu", "precision": "float64"}


def test_the_lives_on_the_cpu_refuse_what_the_torch_lives_and_a_lone_brain_refuse():
    """Every refused call raises the same kind of exception from the lives on the CPU, from
    the lives on a torch device and, where a lone brain has the call, from one brain of as many
    streams."""
    calls = {name: (how, refused, kind) for name, (how, refused, _, kind) in REFUSED.items()}
    calls.update(AT_THE_BOUNDARY)
    for name, (how, refused, kind) in calls.items():
        x = rows(1)
        for place in ({}, PLACE):
            lives = some_lives(**place)
            prepared(lives, how)
            with pytest.raises(kind):
                refused(lives, x)
        if "live" in name:
            continue  # a lone brain's live follows one stream
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lone = cd.Brain.compose(INPUTS, ACTIONS, seed=0, backend="cpu", **OPTIONS)
        prepared(lone, how)
        with pytest.raises(kind):
            refused(SimpleNamespace(act=lone.act, step=lone.step, lesson=lone.lesson,
                                    learn=lone.learn, values=lone.value), x)
