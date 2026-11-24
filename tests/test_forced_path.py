"""The forced alignment reads as its transcript, or the transcript is refused."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from cadence.speech.align import Alignments, even_path, forced_path, soft_path
from cadence.speech.decode import collapse


def paths_by_hand(scores: np.ndarray, sequence: list[int]) -> list[tuple[tuple[int, ...], float]]:
    """Every step-by-step saying of ``sequence`` (CTC's paths) with the sum of its log beliefs."""
    steps, symbols = scores.shape
    out = []
    for said in itertools.product(range(symbols), repeat=steps):
        if collapse(list(said)) == list(sequence):
            out.append((said, float(scores[np.arange(steps), list(said)].sum())))
    return out


def test_a_transcript_the_beliefs_leave_no_path_is_refused():
    scores = np.array([[0.0, -np.inf], [0.0, -np.inf]])  # blank at both steps, never symbol 1
    for states in (False, True):
        with pytest.raises(ValueError, match="leave the transcript no path"):
            forced_path(scores, [1], states=states)
    held = Alignments()
    with pytest.raises(ValueError, match="leave the transcript no path"):
        held.realign("a", scores, [1])
    assert "a" not in held  # no path of blanks is kept for a lesson


def test_a_path_through_beliefs_of_zero_reads_as_its_transcript_and_is_the_best():
    rng = np.random.default_rng(10)
    refused = returned = 0
    for sequence in ([1, 2], [2, 2], [3], [1, 2, 1], [2, 2, 2]):
        lattice = np.zeros(2 * len(sequence) + 1, dtype=np.int64)
        lattice[1::2] = sequence
        for _ in range(40):
            scores = np.log(rng.dirichlet(np.ones(4), size=5))  # 5 steps, blank and three symbols
            scores[rng.random(scores.shape) < 0.45] = -np.inf  # steps a symbol is never said at
            best = max(score for _, score in paths_by_hand(scores, sequence))
            if best == -np.inf:  # every saying of the transcript passes a belief of zero
                with pytest.raises(ValueError, match="leave the transcript no path"):
                    forced_path(scores, sequence)
                refused += 1
                continue
            path = forced_path(scores, sequence)
            assert collapse(path) == sequence
            assert scores[np.arange(5), path].sum() == pytest.approx(best)
            states = forced_path(scores, sequence, states=True)
            assert lattice[states].tolist() == path.tolist()
            assert states[0] <= 1 and states[-1] >= len(lattice) - 2
            assert set(np.diff(states).tolist()) <= {0, 1, 2}
            returned += 1
    assert refused >= 20 and returned >= 20  # both were read


def test_sure_beliefs_give_their_path_and_refuse_every_other_transcript():
    truth = [0, 3, 3, 0, 5, 5, 5, 0, 5, 0]  # a blank parts the two 5s
    with np.errstate(divide="ignore"):
        sure = np.log(np.eye(8)[truth])  # a belief of zero in every other symbol
    assert forced_path(sure, [3, 5, 5]).tolist() == truth
    assert forced_path(sure, [3, 5, 5], states=True).tolist() == [0, 1, 1, 2, 3, 3, 3, 4, 5, 6]
    for other in ([3, 5], [3, 5, 5, 5], [5, 3, 5], [3]):
        with pytest.raises(ValueError, match="leave the transcript no path"):
            forced_path(sure, other)


@pytest.mark.parametrize(
    ("transcript", "message"),
    [
        ([1.5, 2], "each a whole number from zero"),
        ([-1], "each a whole number from zero"),
        ([True, False, True], "each a whole number from zero"),
        ([float("nan")], "each a whole number from zero"),
        (["a"], "each a whole number from zero"),
        ([[1, 2]], "a sequence of symbols"),
        ([8], "outside the beliefs' 8 symbols"),
        ([1, 0, 2], "holds the blank"),
    ],
    ids=["fraction", "negative", "flags", "nan", "text", "nested", "beyond", "blank"],
)
def test_a_transcripts_symbols_are_whole_numbers_among_the_beliefs(transcript, message):
    flat = np.zeros((6, 8))
    with pytest.raises(ValueError, match=message):
        forced_path(flat, transcript)
    with pytest.raises(ValueError, match=message):
        soft_path(flat, transcript)
    whole = forced_path(flat, [1, 2])
    assert collapse(whole) == [1, 2]
    assert forced_path(flat, np.array([1.0, 2.0])).tolist() == whole.tolist()  # whole, as floats
    np.testing.assert_array_equal(soft_path(flat, np.array([1.0, 2.0])), soft_path(flat, [1, 2]))


@pytest.mark.parametrize("blank", [-1, 8, 1.5, True], ids=["negative", "beyond", "fraction", "flag"])
def test_the_blank_is_one_of_the_beliefs_symbols(blank):
    flat = np.zeros((6, 8))
    for path in (forced_path, soft_path):
        with pytest.raises(ValueError, match="blank must be one of the beliefs' 8 symbols"):
            path(flat, [1, 2], blank=blank)
    last = forced_path(flat, [1, 0], blank=7)  # another blank: symbol 0 is then a letter
    assert collapse(last, blank=7) == [1, 0]
    held = Alignments(blank=blank)
    with pytest.raises(ValueError, match="blank must be one of the beliefs' 8 symbols"):
        held.realign("a", flat, [1, 2])
    assert len(held) == 0


@pytest.mark.parametrize("value", [float("nan"), float("inf")], ids=["nan", "infinity"])
def test_beliefs_that_are_no_numbers_below_infinity_are_refused(value):
    scores = np.zeros((6, 8))
    scores[2, 1] = value
    with pytest.raises(ValueError, match="without NaN or positive infinity"):
        forced_path(scores, [1, 2])
    scores[2, 1] = -np.inf  # a belief of zero is a belief
    assert collapse(forced_path(scores, [1, 2])) == [1, 2]


def test_the_even_path_takes_whole_symbols_alone():
    for transcript in ([1.5, 2], [-1, 2], [True, False]):
        with pytest.raises(ValueError, match="each a whole number from zero"):
            even_path(4, transcript)
    assert even_path(4, np.array([1.0, 2.0])).tolist() == [1, 1, 2, 2]
    held = Alignments()
    with pytest.raises(ValueError, match="each a whole number from zero"):
        held.even("a", 4, [1.5, 2])
    assert len(held) == 0
