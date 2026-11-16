"""A lesson given to ``live`` (``lesson=(targets, weights)``): taught at its place in the moment,
after the preceding action's outcome is taken and before the moment's act, so a lesson a world
gives every moment is taken where ``lesson`` before ``live`` is refused while a sampled action
awaits its outcome."""

import copy

import numpy as np
import pytest

import cadence as cd

OBSERVATIONS = np.eye(6)[:, :6]


def brain(*, youth: int = 100) -> cd.Brain:
    return cd.Brain.compose(
        6, 3, modules=(8,), seed=5, recall=True, history_decays=(0.5,), maturity=True,
        arousal={"youth": youth},
    )


def shown(action: int) -> np.ndarray:
    targets = np.zeros((1, 3))
    targets[0, action] = 1.0
    return targets


def archive(held: cd.Brain, path):
    with np.load(held.save(path), allow_pickle=False) as saved:
        return {name: saved[name].copy() for name in saved.files}


def same_archives(left, right):
    assert left.keys() == right.keys()
    for name in left:
        np.testing.assert_array_equal(left[name], right[name], err_msg=name)


def test_a_lesson_through_live_is_taken_where_a_lesson_before_live_is_refused():
    body = brain()  # aroused through its youth: every act is sampled
    body.live(OBSERVATIONS[[0]])
    for t in range(1, 6):
        assert body.basal_ganglia._pending is not None
        with pytest.raises(RuntimeError, match="awaits its outcome"):
            copy.deepcopy(body).lesson(OBSERVATIONS[[t]], targets=shown(t % 3), weights=[0.5])
        twin = copy.deepcopy(body)
        before = body.learner.updates
        body.live(OBSERVATIONS[[t]], reward=[0.2], lesson=(shown(t % 3), 0.5))
        twin.live(OBSERVATIONS[[t]], reward=[0.2])
        # the outcome's step and the lesson's, against the outcome's alone
        assert body.learner.updates == before + 2 == twin.learner.updates + 1
        assert body.learner.contrast_updates == twin.learner.contrast_updates + 1
        assert not np.array_equal(body.brain.efficacy, twin.brain.efficacy)


def test_a_first_moment_s_lesson_is_the_lesson_before_live(tmp_path):
    body, twin = brain(), brain()
    targets = shown(2)
    action = body.live(OBSERVATIONS[[3]], lesson=(targets, [0.75]))
    twin.lesson(OBSERVATIONS[[3]], targets=targets, weights=[0.75])
    assert np.array_equal(twin.live(OBSERVATIONS[[3]]), action)
    same_archives(archive(body, tmp_path / "body"), archive(twin, tmp_path / "twin"))


def test_a_weight_of_one_teaches_as_no_weight(tmp_path):
    body, twin = brain(youth=0), brain(youth=0)
    for held, weights in ((body, None), (twin, [1.0])):
        held.live(OBSERVATIONS[[0]])
        held.live(OBSERVATIONS[[1]], reward=[0.5], lesson=(shown(1), weights))
    same_archives(archive(body, tmp_path / "body"), archive(twin, tmp_path / "twin"))


@pytest.mark.parametrize("lesson", [
    (np.full((1, 3), 0.5), None),  # the targets sum to more than one
    (shown(0), [-1.0]),
    (shown(0), [0.5, 0.5]),
    (np.eye(4)[:1], None),
    (shown(0),),  # not a pair
    shown(0)[0],
])
def test_an_invalid_lesson_refuses_before_any_live_state_changes(tmp_path, lesson):
    body = brain()
    body.live(OBSERVATIONS[[0]])
    decision = body.decision_id
    before = archive(body, tmp_path / "before")
    with pytest.raises(ValueError):
        body.live(OBSERVATIONS[[1]], reward=[0.0], decision_id=decision, lesson=lesson)
    assert body.decision_id == decision and body.pending_feedback
    same_archives(before, archive(body, tmp_path / "after"))
