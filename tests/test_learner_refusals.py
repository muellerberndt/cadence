"""The learner refuses a malformed drive, and a warm state of another batch, as the solver
refuses them, whatever the labels: a batch whose caller-read answer agrees with every label
(which runs no phase) is refused as a batch that differs, on the host and for a drive or a state
that rests on a device, and nothing changes. A valid batch that agrees still runs no phase."""

from __future__ import annotations

import numpy as np
import pytest
from places import PLACES, place

import cadence as cd

LABELS = np.array([0, 0])
ANSWERS = [
    pytest.param(np.array([0, 0]), id="every row agrees"),
    pytest.param(np.array([1, 0]), id="one row differs"),
    pytest.param(np.array([1, 2]), id="every row differs"),
    pytest.param(None, id="judged by the free phase"),
]


def pupil(**settings) -> cd.Brain:
    return cd.Brain.compose(8, 3, modules=(8,), seed=2, learning_surprise=True,
                            **{"backend": "cpu", **settings})


def malformed(neurons: int) -> list[tuple[str, np.ndarray]]:
    one = np.zeros((2, neurons))
    one[1, 3] = np.inf
    return [("one column", np.ones((2, 1))), ("too wide", np.ones((2, neurons + 1))),
            ("not a number", np.full((2, neurons), np.nan)), ("infinite", one)]


@pytest.mark.parametrize("answer", ANSWERS)
def test_a_malformed_drive_is_refused_as_the_solver_refuses_it(answer):
    brain = pupil()
    learner, graph = brain.learner, brain.brain
    before = (graph.efficacy.copy(), graph.bias.copy(), learner.updates)
    for name, drive in malformed(brain.connectome.n):
        with pytest.raises(ValueError) as solver:
            graph.settle_batch(drive, steps=1)
        with pytest.raises(ValueError) as lesson:
            learner.step(drive, LABELS, answer=answer)
        assert str(lesson.value) == str(solver.value), name
    np.testing.assert_array_equal(graph.efficacy, before[0])
    np.testing.assert_array_equal(graph.bias, before[1])
    assert learner.updates == before[2]


@pytest.mark.parametrize("answer", ANSWERS)
@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_a_malformed_drive_on_a_device_is_refused_as_the_solver_refuses_it(answer, precision):
    torch = pytest.importorskip("torch")
    brain = pupil(backend="torch", device="cpu", precision=precision)
    learner, graph = brain.learner, brain.brain
    before = (graph.efficacy.copy(), graph.bias.copy(), learner.updates)
    kind = torch.float64 if precision == "float64" else torch.float32
    for name, drive in malformed(brain.connectome.n):
        resident = torch.as_tensor(drive, dtype=kind)
        with pytest.raises(ValueError) as solver:
            graph.settle_batch(resident, steps=1)
        with pytest.raises(ValueError) as lesson:
            learner.step(resident, LABELS, answer=answer)
        assert str(lesson.value) == str(solver.value), name
    np.testing.assert_array_equal(graph.efficacy, before[0])
    np.testing.assert_array_equal(graph.bias, before[1])
    assert learner.updates == before[2]


@pytest.mark.parametrize("answer", ANSWERS)
@pytest.mark.parametrize("where", PLACES)
def test_a_warm_state_of_another_batch_is_refused_as_the_solver_refuses_it(answer, where):
    # a lesson judged by its answer takes the warm state's rows: a state of fewer rows raised
    # IndexError on the host, and one of more rows resting on a device was read in part
    brain = pupil(**place(where))
    learner, graph = brain.learner, brain.brain
    before = (np.array(graph.efficacy), np.array(graph.bias), learner.updates)
    drive = brain.stimulus(np.eye(8)[[0, 1]], memory=False)
    for rows in (1, 3):
        warm = graph.settle_batch(brain.stimulus(np.eye(8)[:rows], memory=False), steps=2)
        with pytest.raises(ValueError) as solver:
            graph.settle_batch(drive, steps=1, state=warm)
        with pytest.raises(ValueError) as lesson:
            learner.step(drive, LABELS, warm, answer=answer)
        assert str(lesson.value) == str(solver.value), rows
    np.testing.assert_array_equal(graph.efficacy, before[0])
    np.testing.assert_array_equal(graph.bias, before[1])
    assert learner.updates == before[2] and learner.contrast_updates == 0


def test_a_batch_of_another_size_than_its_labels_is_refused_first():
    brain = pupil()
    with pytest.raises(ValueError, match="same batch size"):
        brain.learner.step(np.full((3, brain.connectome.n), np.nan), LABELS, answer=LABELS)


def test_a_valid_batch_that_agrees_runs_no_phase(monkeypatch):
    brain = pupil()
    drive = brain.stimulus(np.eye(8)[[0, 1]], memory=False)

    def settle(*args, **kwargs):
        raise AssertionError("a lesson of no row settles nothing")

    monkeypatch.setattr(type(brain.brain), "settle_batch", settle)
    monkeypatch.setattr(type(brain.brain), "equilibrate", settle)
    learned, report = brain.learner.step(drive, LABELS, answer=LABELS)
    assert learned.taught is not None and not learned.taught.any()
    assert report["total_steps"] == 0.0 and report["agreed_presentations"] == 2.0
    assert brain.learner.updates == 0
