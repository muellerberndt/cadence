"""Lessons from transcripts: the forced alignment of the brain's own beliefs."""

import numpy as np
import pytest

import cadence as cd
from cadence.speech.align import Alignments, even_path, forced_path, log_beliefs
from cadence.speech.decode import collapse


def test_the_even_path_spreads_the_symbols():
    assert even_path(6, [3, 5]).tolist() == [3, 3, 3, 5, 5, 5]
    assert even_path(5, [1, 2, 3]).tolist() == [1, 1, 2, 2, 3]


def test_the_forced_path_follows_the_beliefs_and_parts_doubled_letters():
    truth = [0, 3, 3, 0, 5, 5, 5, 0, 5, 0]  # "c" then "e", "e": a blank parts the two e's
    scores = np.log(np.full((10, 8), 0.02))
    scores[np.arange(10), truth] = 0.0
    assert forced_path(scores, [3, 5, 5]).tolist() == truth
    assert forced_path(scores, [3, 5, 5], states=True).tolist() == [0, 1, 1, 2, 3, 3, 3, 4, 5, 6]
    # with no beliefs at all every symbol still takes a step, and the doubled letter stays two
    flat = forced_path(np.zeros((6, 8)), [2, 2])
    assert collapse(flat) == [2, 2]
    with pytest.raises(ValueError):
        forced_path(np.zeros((2, 8)), [2, 2])  # two e's need three steps


def test_an_utterance_heard_twice_keeps_a_path_for_each_hearing():
    held = Alignments()
    held.even((0, 7), 4, [1, 2])  # utterance 7 in pass 0
    held.realign((1, 7), np.log(np.eye(4)[[1, 1, 1, 2]] + 1e-3), [1, 2])  # and in pass 1
    first = held.take((0, 7))
    assert first.tolist() == [1, 1, 2, 2] and (1, 7) in held  # taking one keeps the other
    labels = held.lesson([(1, 7)], 6)
    assert labels[:, 0].tolist() == [1, 1, 1, 2, 0, 0]  # blank after the hearing's end
    assert len(held) == 0
    with pytest.raises(KeyError):
        held.take((1, 7))


def test_a_brains_beliefs_are_read_side_by_side_from_rest():
    brain = cd.Brain.compose(6, 5, seed=0, backend="cpu", recall=False)
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(4, 6)), rng.normal(size=(2, 6))
    both = log_beliefs(brain, [a, b])
    assert [x.shape for x in both] == [(4, 5), (2, 5)]
    np.testing.assert_allclose(np.exp(both[0]).sum(axis=1), 1.0)
    alone = log_beliefs(brain, [b])[0]
    np.testing.assert_allclose(alone, both[1], atol=1e-9)  # streams do not mix
