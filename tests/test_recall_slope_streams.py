"""The recall memory's slope rule reads each stream's own slots (#292): a stream's slope lesson
does not depend on how long another stream of the bank is."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import cadence as cd

INPUTS, ACTIONS = 12, 4
rng = np.random.default_rng(0)
MOMENTS = [rng.random((2, INPUTS)) for _ in range(41)]
LABELS = np.array([1, 2])


def slope_step(rows, **place):
    """The slope maps' largest change in one lesson after 40 moments of the streams ``rows``.
    Stream 1 begins again at moment 36. Every reward is zero, so the lesson is the first thing
    the brain learns (the reproduction of #292)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        brain = cd.Brain.compose(INPUTS, ACTIONS, seed=0, episodic=False,
                                 learning_surprise=False, **({"backend": "cpu"} | place))
        recall = brain.recall
        for t in range(40):
            done = np.array([row == 1 and t == 36 for row in rows])
            outcome = {} if t == 0 else {"reward": np.zeros(len(rows)), "done": done}
            brain.step(MOMENTS[t][rows], **outcome)
        own = recall.held.sum(axis=1).tolist()
        before = [np.array(part) for part in recall.slope_maps[0]]
        brain.step(MOMENTS[40][rows], reward=np.zeros(len(rows)), teacher=LABELS[rows])
        after = [np.array(part) for part in recall.slope_maps[0]]
    return own, max(float(np.abs(a - b).max()) for a, b in zip(after, before, strict=True))


def test_a_stream_within_the_table_steps_the_slope_beside_one_past_it():
    both = slope_step([0, 1])
    alone = slope_step([1])
    past = slope_step([0])
    assert both[0] == [41, 5] and alone[0] == [5] and past[0] == [41]
    assert past[1] == 0.0  # a stream past the table leans nowhere, alone or not
    assert alone[1] > 1e-3
    assert both[1] > 1e-4  # the short stream's lean, in the mean over the lesson's two rows


def lived(streams: int, ends, moments: int = 60, **options):
    """A brain of ``streams`` streams living ``moments`` with a lesson at every third moment;
    ``ends(t)`` gives the moment's endings. Returns whether every lesson's streams took the slope
    rule the bank's width gives (``Recall._slope_as_bank``)."""
    agree = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        brain = cd.Brain.compose(INPUTS, ACTIONS, seed=1, backend="cpu", **options)
        recall = brain.recall
        feed = np.random.default_rng(2)
        for t in range(moments):
            x = feed.random((streams, INPUTS))
            outcome = {} if t == 0 else {"reward": feed.random(streams), "done": ends(t)}
            teacher = feed.integers(ACTIONS, size=streams) if t % 3 == 1 else None
            if teacher is not None and recall.slots:
                agree.append(recall._slope_as_bank(np.arange(streams)))
            brain.step(x, teacher=teacher, **outcome)
    return agree


def test_one_stream_reads_the_rule_of_its_bank_at_every_lesson():
    """One stream's bank is as wide as its own slots: every lesson reads the rule it read on
    0.96.1, so its bits are kept."""
    ending = np.random.default_rng(3)
    agree = lived(1, lambda t: ending.random(1) < 0.05)
    assert agree and all(agree)


def test_streams_of_equal_length_read_the_rule_of_their_bank_at_every_lesson():
    agree = lived(3, lambda t: np.full(3, t % 25 == 0))
    assert agree and all(agree)


def test_streams_of_unequal_length_part_from_the_banks_rule_past_the_table():
    agree = lived(2, lambda t: np.array([False, t == 45]))
    assert not all(agree)  # stream 1 begins again at 45 while stream 0 runs past the table


def test_the_compiled_lesson_takes_the_same_step_as_the_hosts():
    pytest.importorskip("numba")
    host = slope_step([0, 1])
    compiled = slope_step([0, 1], learning_compiled_moment=True)
    assert compiled[1] == pytest.approx(host[1], rel=1e-9)


torch = pytest.importorskip("torch")


def test_the_device_path_reads_each_streams_own_slots(monkeypatch):
    monkeypatch.setenv("CADENCE_TORCH_MEMORY", "1")
    place = {"backend": "torch", "device": "cpu", "precision": "float64"}
    host = slope_step([0, 1])
    device = slope_step([0, 1], **place)
    assert device[1] == pytest.approx(host[1], rel=1e-6)
