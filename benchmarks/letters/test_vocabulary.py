"""The closed word list and the list-matched reading, without the corpus."""

from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path

_here = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("vocabulary", _here / "vocabulary.py")
assert _spec is not None and _spec.loader is not None
vc = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("vocabulary", vc)
_spec.loader.exec_module(vc)


def test_the_list_is_the_most_frequent_words() -> None:
    total = Counter(vc.words_of("the cat saw the dog; THE dog's cat"))
    assert vc.top(total, 3) == ["the", "cat", "dog"]
    assert vc.on_list("The cat", {"the", "cat"})
    assert not vc.on_list("the bird", {"the", "cat"})


def test_said_words_match_the_nearest_list_word() -> None:
    match = vc.Matcher(["the", "short", "shore", "sort"])
    assert match("short") == "short"
    assert match("shorrk") == "short"  # two edits to short and to shore: the more frequent
    assert vc.distance("kitten", "sitting") == 3


def test_words_read_open_and_matched() -> None:
    match = vc.Matcher(["the", "short", "day"])
    out = vc.read_words(["the shorrk day"], ["the short day"], match)
    assert out["words_open"] == 1 - 1 / 3 and out["words_matched"] == 1.0
    assert out["exact_open"] == 0.0 and out["exact_matched"] == 1.0


def test_the_list_distances_are_the_word_distances() -> None:
    listed = ["the", "short", "shore", "sort", "a", "day", "they're"]
    match = vc.Matcher(listed)
    for word in ["shorrk", "", "a", "theyre", "dai", "zzzzzzzzzz"]:
        assert match.distances(word).tolist() == [vc.distance(word, w) for w in listed]
    assert match.place("shorrk", "short") == 0 and match.place("shorrk", "shore") == 1
    assert match.place("zz", "the") == 1  # behind "a", which is one edit nearer


def test_candidates_place_each_reference_word() -> None:
    match = vc.Matcher(["the", "short", "shore", "day", "dog"])
    assert vc.align(["a", "b", "c"], ["a", "c"]) == [(0, 0), (1, None), (2, 1)]
    out = vc.candidates(["the shorrk dai extra"], ["the shore day dog"], match, within=(1, 2))
    # the: right; shore: second among the nearest of shorrk; day: nearest of dai; dog: further
    assert out["reference_words"] == 4
    assert out["right"] == 0.25 and out["within_1"] == 0.25 and out["within_2"] == 0.25
    assert out["further"] == 0.25 and out["dropped"] == 0.0
    assert out["right"] + out["within_1"] + out["within_2"] + out["further"] + out["dropped"] == 1.0
