"""A surprise lesson judges its rows by the settle the act on the same moment reads. When it
teaches no row nothing the settle reads has changed, so the act reads the lesson's settle: the
same state and the same report, without a second solve."""

from __future__ import annotations

import copy
import dataclasses

import numpy as np
import pytest

import cadence as cd
from cadence.brain import Brain as NeuralGraph


def composed(**options) -> cd.Brain:
    return cd.Brain.compose(6, 3, modules=(16,), seed=0, recall=True, learning_surprise=True,
                            teaching="normalized", episodic=False, **options)


def agreed(brain: cd.Brain, x: np.ndarray) -> np.ndarray:
    """The labels the brain would answer at this moment: a lesson on them teaches no row."""
    twin = copy.deepcopy(brain)
    return twin.act(x, greedy=True)


def moments(brain: cd.Brain, count: int = 3) -> tuple[np.ndarray, np.ndarray]:
    x = np.eye(6)[np.array([[0, 1], [2, 3], [4, 5], [1, 2]])]
    for t in range(count):
        brain.act(x[t], greedy=True)
    return x[count], agreed(brain, x[count])


@pytest.fixture()
def settles(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls: list[int] = []
    original = NeuralGraph.equilibrate

    def counted(self: NeuralGraph, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(NeuralGraph, "equilibrate", counted)
    return calls


def test_the_act_after_a_lesson_that_taught_no_row_reads_its_settle(settles: list[int]) -> None:
    brain = composed()
    plain = copy.deepcopy(brain)
    x, labels = moments(brain)
    moments(plain)
    report = brain.lesson(x, labels)
    assert report["accepted_presentations"] == 0
    settled = len(settles)
    action = brain.act(x, greedy=True)
    assert len(settles) == settled  # no solve of its own
    # the same as an act that solves: state, report, action
    alone = plain.act(x, greedy=True)
    np.testing.assert_array_equal(action, alone)
    np.testing.assert_array_equal(brain.basal_ganglia.state.v, plain.basal_ganglia.state.v)
    np.testing.assert_array_equal(brain.basal_ganglia.state.activation,
                                  plain.basal_ganglia.state.activation)
    assert dict(brain.last_settlement) == dict(plain.last_settlement)
    # it is read once: the next act solves
    brain.act(x, greedy=True)
    assert len(settles) > settled


def test_a_lesson_that_taught_a_row_is_settled_again(settles: list[int]) -> None:
    brain = composed()
    x, labels = moments(brain)
    report = brain.lesson(x, (labels + 1) % 3)
    assert report["accepted_presentations"] == 2
    settled = len(settles)
    brain.act(x, greedy=True)
    assert len(settles) > settled


@pytest.mark.parametrize("change", ["efficacy", "bias", "state", "config", "observation"])
def test_what_the_settle_reads_changing_settles_again(
    settles: list[int], change: str
) -> None:
    brain = composed()
    x, labels = moments(brain)
    brain.lesson(x, labels)
    if change == "efficacy":
        brain.brain.efficacy = brain.brain.efficacy * 0.9
    elif change == "bias":
        brain.brain.bias = brain.brain.bias + 0.01
    elif change == "state":
        brain.basal_ganglia.state.v[:] += 0.01
    elif change == "config":
        brain.retune(temperature=0.3)
    else:
        x = x[::-1]
    settled = len(settles)
    brain.act(x, greedy=True)
    assert len(settles) > settled


def test_a_copy_and_a_checkpoint_taken_after_the_lesson_continue_alike(tmp_path) -> None:
    brain = composed()
    x, labels = moments(brain)
    brain.lesson(x, labels)
    twin = copy.deepcopy(brain)
    path = brain.save(tmp_path / "brain.npz")
    loaded = cd.Brain.load(path)
    action = brain.act(x, greedy=True)
    for other in (twin, loaded):
        np.testing.assert_array_equal(other.act(x, greedy=True), action)
        np.testing.assert_array_equal(other.basal_ganglia.state.v, brain.basal_ganglia.state.v)
        assert dict(other.last_settlement) == dict(brain.last_settlement)


def test_a_refused_judge_is_not_kept() -> None:
    brain = composed()
    x, labels = moments(brain)
    brain.learner.config = dataclasses.replace(
        brain.learner.config, free_steps=0, tolerance=1e-30
    )
    with pytest.raises(RuntimeError):
        brain.lesson(x, labels)
    assert brain._judged is None
