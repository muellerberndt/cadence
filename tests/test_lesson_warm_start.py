"""A lesson's free phase starts where the act settles: warm from the stream's activity, rest after
``reset`` and for streams that have not acted. In a deep reciprocal brain one observation can hold
two fixed points, one reached warm from the stream and one from rest; a lesson that started from
rest taught at the rest one while the act answered at the warm one, and lessons never corrected
the act."""

from __future__ import annotations

import copy
import itertools
import warnings

import numpy as np
import pytest

import cadence as cd

X = np.eye(8)


def parameters(brain: cd.Brain) -> np.ndarray:
    return np.concatenate([brain.brain.efficacy, brain.brain.bias])


def taught_from_rest(brain: cd.Brain, *args, **kwargs) -> dict[str, float]:
    """The lesson as 0.95.1 gave it: its free phase from rest."""
    brain._activity = lambda: None  # type: ignore[method-assign]
    try:
        return brain.lesson(*args, **kwargs)
    finally:
        del brain._activity


def two_fixed_points() -> tuple[cd.Brain, int, int]:
    """A deep reciprocal brain (two regions, qualified solves) taught one-hot cues, and a cue pair
    ``(before, cue)`` where the cue's drive after ``before`` settles warm to a fixed point whose
    answer differs from the one rest reaches under the same drive."""
    for seed, modules in itertools.product(range(4), [(48, 24), (32, 16, 8)]):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            brain = cd.Brain.compose(
                8, 4, modules=modules, seed=seed, recall=False, learning_qualified=True,
                learning_nudged_steps=1024,
            )
            labels = np.random.default_rng(seed).integers(0, 4, 8)
            try:
                for _ in range(200):
                    brain.lesson(X, labels)
            except cd.LearningPhaseError:
                continue
        for before, cue in itertools.permutations(range(8), 2):
            brain.reset()
            brain.act(X[[before]], greedy=True)
            # the cue's drive after ``before``, settled warm from the stream and from rest
            brain._prepare(X[[cue]])
            drive = brain._moment_drive(X[[cue]])
            brain._prepared = None
            brain.basal_ganglia._drive = None
            try:
                warm = brain.learner.free(drive, brain._activity())
                rest = brain.learner.free(drive)
            except cd.LearningPhaseError:
                continue
            answers = int(brain.learner._answer(warm)[0]), int(brain.learner._answer(rest)[0])
            if answers[0] != answers[1]:
                brain.reset()
                return brain, before, cue
    raise AssertionError("no brain of the search holds two fixed points for one drive")


def test_a_lesson_corrects_the_act_where_rest_reaches_another_fixed_point():
    brain, before, cue = two_fixed_points()

    def acted(b: cd.Brain) -> tuple[int, int]:
        b.reset()
        b.act(X[[before]], greedy=True)
        warm = int(b.act(X[[cue]], greedy=True)[0])
        b.reset()
        rest = int(b.act(X[[cue]], greedy=True)[0])
        b.reset()
        return warm, rest

    warm, rest = acted(brain)
    assert warm != rest
    label = np.array([rest])  # the answer rest gives; the act, warm from the stream, misses it
    library, rested = copy.deepcopy(brain), copy.deepcopy(brain)
    for _ in range(10):
        for b, lesson in ((library, cd.Brain.lesson), (rested, taught_from_rest)):
            b.reset()
            b.act(X[[before]], greedy=True)
            lesson(b, X[[cue]], label)
    assert acted(library)[0] == rest  # the lesson reached the act's fixed point and corrected it
    assert acted(rested)[0] == warm  # from rest the lessons found the answer given, and taught little


@pytest.mark.parametrize("modules", [(16,), (16, 8)])
def test_the_lesson_starts_from_the_state_the_act_settles_from(monkeypatch, modules):
    brain = cd.Brain.compose(12, 5, modules=modules, seed=3)
    x = np.eye(12)[[1, 7]]
    brain.act(np.eye(12)[[2, 5]], greedy=True)
    stream = brain.basal_ganglia.state
    seen = []
    step = brain.learner.step

    def spied(drive, labels, warm=None, *args, **kwargs):
        seen.append(warm)
        return step(drive, labels, warm, *args, **kwargs)

    monkeypatch.setattr(brain.learner, "step", spied)
    brain.lesson(x, np.array([0, 3]))
    assert seen[-1] is stream
    brain.reset()
    brain.lesson(x, np.array([0, 3]))
    assert seen[-1] is None


@pytest.mark.parametrize("settings", [{}, {"recall": False}, {"modules": (16, 8)},
                                      {"learning_qualified": True}])
def test_a_lesson_at_rest_is_the_lesson_from_rest_to_the_bit(settings):
    """After ``reset``, and before a stream acted, the lesson is 0.95.1's."""
    settings = {"modules": (16,), **settings}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        founder = cd.Brain.compose(12, 5, seed=3, **settings)
    x, labels = np.eye(12)[[1, 7, 3]], np.array([0, 3, 2])
    founder.act(np.eye(12)[[2, 5, 9]], greedy=True)
    founder.reset()
    library, rested = copy.deepcopy(founder), copy.deepcopy(founder)
    for _ in range(3):
        assert library.lesson(x, labels) == taught_from_rest(rested, x, labels)
    np.testing.assert_array_equal(parameters(library), parameters(rested))
    fresh = cd.Brain.compose(12, 5, seed=3, **settings)
    twin = copy.deepcopy(fresh)
    fresh.lesson(x, labels)
    taught_from_rest(twin, x, labels)
    np.testing.assert_array_equal(parameters(fresh), parameters(twin))


def test_a_lesson_inside_a_step_starts_where_the_act_of_the_step_does(monkeypatch):
    """``step`` learns the outcome, teaches, then acts: the lesson and the act start from one
    state, the stream's activity after the outcome."""
    brain = cd.Brain.compose(12, 5, modules=(16,), seed=3)
    brain.step(np.eye(12)[[1, 2]])
    seen, settled = [], []
    step, qualified = brain.learner.step, brain._qualified

    def spied(drive, labels, warm=None, *args, **kwargs):
        seen.append(warm)
        return step(drive, labels, warm, *args, **kwargs)

    def acted(drive, state=None, **kwargs):
        if kwargs.get("operation") == "act" and not kwargs.get("judges"):
            settled.append(state)
        return qualified(drive, state, **kwargs)

    monkeypatch.setattr(brain.learner, "step", spied)
    monkeypatch.setattr(brain, "_qualified", acted)
    brain.step(np.eye(12)[[3, 4]], reward=np.ones(2), teacher=np.array([0, 1]))
    assert seen and settled and seen[-1] is settled[-1]


def test_a_lesson_on_torch_starts_from_the_state_on_the_device(monkeypatch):
    pytest.importorskip("torch")
    brain = cd.Brain.compose(12, 5, modules=(16,), seed=3, backend="torch", device="cpu")
    host = cd.Brain.compose(12, 5, modules=(16,), seed=3, backend="cpu", learning_fast_settle=False)
    torch_brain = cd.Brain.compose(
        12, 5, modules=(16,), seed=3, backend="torch", device="cpu", learning_fast_settle=False
    )
    seen = []
    step = brain.learner.step

    def spied(drive, labels, warm=None, *args, **kwargs):
        seen.append(warm)
        return step(drive, labels, warm, *args, **kwargs)

    monkeypatch.setattr(brain.learner, "step", spied)
    brain.act(np.eye(12)[[2, 5]], greedy=True)
    brain.lesson(np.eye(12)[[1, 7]], np.array([0, 3]))
    assert seen[-1] is not None and seen[-1].device is not None  # warm, and resting on the device
    for b in (host, torch_brain):
        b.act(np.eye(12)[[2, 5]], greedy=True)
        b.lesson(np.eye(12)[[1, 7]], np.array([0, 3]))
    np.testing.assert_allclose(parameters(torch_brain), parameters(host), rtol=0, atol=1e-9)


def test_a_learned_code_keeps_its_phase_in_the_warm_lesson(monkeypatch):
    brain = cd.Brain.compose(16, 4, modules=(16,), seed=1, learned_code=8, recall=False)
    rng = np.random.default_rng(0)
    brain.act((rng.random((2, 16)) < 0.3).astype(float), greedy=True)
    stream = brain.basal_ganglia.state
    seen = []
    step = brain.learner.step

    def spied(drive, labels, warm=None, *args, **kwargs):
        seen.append(warm)
        return step(drive, labels, warm, *args, **kwargs)

    monkeypatch.setattr(brain.learner, "step", spied)
    brain.lesson((rng.random((2, 16)) < 0.3).astype(float), np.array([0, 3]))
    warm = seen[-1]
    assert warm is not None and warm is not stream  # the stream's state with the code's phase in it
    others = np.setdiff1d(
        np.arange(brain.connectome.n),
        np.concatenate([brain.sensory_index, brain.prefrontal_index, brain.code_index]),
    )
    np.testing.assert_array_equal(
        np.asarray(warm.v)[:, others], np.asarray(stream.v)[:, others]
    )
