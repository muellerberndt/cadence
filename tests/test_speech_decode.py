"""Reading text: the collapse of said symbols and the beam over a word list's trie."""

import numpy as np
import pytest

from cadence.speech.decode import Alphabet, WordBeam, collapse, greedy

AB = Alphabet()


def beliefs(said, *, sure=0.9):
    """Log beliefs that say ``said`` (one symbol a step) with weight ``sure``."""
    out = np.full((len(said), AB.size), (1 - sure) / (AB.size - 1))
    out[np.arange(len(said)), said] = sure
    return np.log(out)


def test_the_alphabet_round_trips_text():
    symbols = AB.encode("Don't  stop, now")
    assert AB.decode(symbols) == "don't stop now"
    assert AB.size == 29 and AB.space == 28
    assert symbols.tolist()[:5] == [4, 15, 14, 27, 20]


def test_collapse_reads_repeats_once_and_drops_blanks():
    assert collapse([0, 3, 3, 0, 3, 1, 1, 0]) == [3, 3, 1]
    assert collapse([]) == []
    c, a, t, s = (AB.encode(x)[0] for x in "cats")
    assert greedy(beliefs([c, c, 0, a, t, t, 28, s])) == "cat s"


def test_the_beam_writes_only_list_words():
    c, a, t, r, o = (int(AB.encode(x)[0]) for x in "catro")
    # said: "cat ro" (as letters, no blanks); the list holds "cat" and "rot"
    said = beliefs([c, a, t, 28, r, o])
    assert greedy(said) == "cat ro"
    beam = WordBeam(["cat", "rot"], beam=8)
    # "rot" needs a t the beliefs hold at .1/28 on its own step: the open word is dropped
    assert beam.decode(said) == "cat"
    # with one more step that leans to t, the beam completes the list word
    said = np.vstack([said, beliefs([t], sure=0.4)])
    assert beam.decode(said) == "cat rot"


def test_the_beam_picks_the_list_word_the_beliefs_hold_most():
    c, a, t, r = (int(AB.encode(x)[0]) for x in "catr")
    # step 3 holds t at .45 and r at .5: greedy says "car", the list knows only "cat"
    rows = beliefs([c, a, r])
    rows[2] = np.log(np.full(AB.size, 0.05 / 27))
    rows[2, r], rows[2, t] = np.log(0.5), np.log(0.45)
    assert greedy(rows) == "car"
    assert WordBeam(["cat", "cab"]).decode(rows) == "cat"
    assert WordBeam(["cat", "car"]).decode(rows) == "car"


def test_doubled_letters_need_a_blank_between():
    b, e = (int(AB.encode(x)[0]) for x in "be")
    held = beliefs([b, e, e])  # e held over two steps: one e
    parted = beliefs([b, e, 0, e])  # e, blank, e: two e's
    beam = WordBeam(["be", "bee"])
    assert beam.decode(held) == "be"
    assert beam.decode(parted) == "bee"


def test_counts_give_the_more_frequent_word_the_tie():
    c, a, t, r = (int(AB.encode(x)[0]) for x in "catr")
    rows = beliefs([c, a, r])
    rows[2] = np.log(np.full(AB.size, 0.1 / 27))
    rows[2, r] = rows[2, t] = np.log(0.45)
    assert WordBeam(["cat", "car"], counts=[100, 1], prior_weight=1.0).decode(rows) == "cat"
    assert WordBeam(["cat", "car"], counts=[1, 100], prior_weight=1.0).decode(rows) == "car"


def test_a_word_off_the_alphabet_and_wrong_shapes_are_refused():
    with pytest.raises(ValueError):
        WordBeam(["two words"])
    with pytest.raises(ValueError):
        WordBeam(["cat"]).decode(np.zeros((3, 5)))
    with pytest.raises(ValueError):
        WordBeam(["cat"], counts=[1, 2])
