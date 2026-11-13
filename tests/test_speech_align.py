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


def _paths_by_hand(scores, sequence, blank=0):
    """Every step-by-step saying of ``sequence`` (CTC's paths) with the product of its beliefs."""
    import itertools

    steps, symbols = scores.shape
    out = []
    for said in itertools.product(range(symbols), repeat=steps):
        if collapse(list(said)) == list(sequence):
            out.append((said, float(np.exp(scores[np.arange(steps), list(said)].sum()))))
    return out


def test_the_soft_path_is_the_share_of_the_paths_that_say_each_symbol():
    from cadence.speech.align import soft_path

    rng = np.random.default_rng(3)
    scores = np.log(rng.dirichlet(np.ones(4), size=5))  # 5 steps, blank and three symbols
    for sequence in ([1, 2], [2, 2], [3], [1, 2, 1]):
        paths = _paths_by_hand(scores, sequence)
        total = sum(w for _, w in paths)
        by_hand = np.zeros((5, 4))
        for said, w in paths:
            by_hand[np.arange(5), list(said)] += w / total
        soft = soft_path(scores, sequence)
        np.testing.assert_allclose(soft, by_hand, atol=1e-12)
        np.testing.assert_allclose(soft.sum(axis=1), 1.0)


def test_the_soft_path_teaches_the_descent_of_the_transcripts_likelihood():
    from cadence.speech.align import soft_path

    rng = np.random.default_rng(4)
    logits = rng.normal(size=(6, 5))
    sequence = [2, 2, 4]

    def loss(z):
        logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        return -np.log(sum(w for _, w in _paths_by_hand(logp, sequence)))

    logp = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
    pull = soft_path(logp, sequence) - np.exp(logp)  # what a cross-entropy nudge pulls along
    slope = np.zeros_like(logits)
    for index in np.ndindex(*logits.shape):
        moved = logits.copy()
        moved[index] += 1e-5
        back = logits.copy()
        back[index] -= 1e-5
        slope[index] = (loss(moved) - loss(back)) / 2e-5
    np.testing.assert_allclose(pull, -slope, atol=1e-6)


def test_the_soft_path_of_sure_beliefs_is_the_forced_path():
    from cadence.speech.align import soft_path

    truth = [0, 3, 3, 0, 5, 5, 5, 0, 5, 0]
    scores = np.log(np.full((10, 8), 1e-9))
    scores[np.arange(10), truth] = 0.0
    soft = soft_path(scores, [3, 5, 5])
    assert np.argmax(soft, axis=1).tolist() == truth and soft.max(axis=1).min() > 0.999
    flat = soft_path(np.zeros((6, 8)), [2, 2])  # no beliefs: every path alike
    np.testing.assert_allclose(flat.sum(axis=1), 1.0)
    assert flat[:, [0, 2]].sum() == pytest.approx(6.0) and flat[0, 2] > 0 and flat[-1, 0] > 0
    with pytest.raises(ValueError):
        soft_path(np.zeros((2, 8)), [2, 2])  # two e's need three steps
    with pytest.raises(ValueError):
        soft_path(np.full((4, 8), -np.inf), [2])  # beliefs that allow nothing


def test_soft_paths_are_taught_as_targets_side_by_side():
    held = Alignments()
    beliefs = np.log(np.eye(4)[[1, 1, 1, 2]] + 1e-3)
    held.realign("a", beliefs, [1, 2], soft=True)
    held.even("b", 2, [3])
    with pytest.raises(ValueError):
        held.lesson(["a"], 6)  # a soft path is taught as targets
    with pytest.raises(ValueError):
        held.targets(["a"], 6, 5)  # over the symbols its beliefs held
    assert "a" in held and "b" in held  # a refusal takes nothing
    targets = held.targets(["a", "b"], 6, 4)
    assert targets.shape == (6, 2, 4) and len(held) == 0
    np.testing.assert_allclose(targets.sum(axis=2), 1.0)
    assert np.argmax(targets[:, 0], axis=1).tolist() == [1, 1, 1, 2, 0, 0]
    assert targets[:, 1].tolist() == np.eye(4)[[3, 3, 0, 0, 0, 0]].tolist()
    brain = cd.Brain.compose(6, 4, seed=0, backend="cpu", recall=False)
    brain.reset()
    rows = np.random.default_rng(0).normal(size=(6, 2, 6))
    for t in range(6):  # a lesson of distributions, then a greedy act, in place of step
        report = brain.lesson(rows[t], targets=targets[t])
        brain.act(rows[t], greedy=True)
    assert report["accepted_presentations"] >= 1
