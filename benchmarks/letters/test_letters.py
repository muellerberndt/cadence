"""The letters chamber's world and readings, without a download."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("letters", _here / "letters.py")
assert _spec is not None and _spec.loader is not None
lt = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("letters", lt)
_spec.loader.exec_module(lt)

PROTOCOL = _here / "protocol-step0.json"


def test_lexicon_holds_out_words_whose_letters_lessons_teach() -> None:
    lexicon = lt.make_lexicon(3)
    words = lexicon["words"]
    assert len({tuple(w) for w in words}) == len(words) == 40
    assert all(3 <= len(w) <= 5 for w in words)
    assert all(a != b for w in words for a, b in zip(w, w[1:], strict=False))
    taught = {letter for k in lexicon["trained"] for letter in words[k]}
    assert taught == set(range(16))
    assert set(lexicon["held_out"]).isdisjoint(lexicon["trained"])


def test_stream_labels_say_each_letter_once_at_its_sound() -> None:
    lexicon = lt.make_lexicon(0)
    stream = lt.make_stream(lexicon, 30, np.random.default_rng(1))
    assert not any(stream["held_out"])  # training draws trained words only
    y = lt.labels(stream, "onset")
    letters = 16
    space = lt.space_symbol(letters)
    assert lt.text_of(lt.said(y), letters) == stream["text"]
    assert int(np.sum(y == space)) == 30
    for symbol, start, end in stream["sounds"]:
        assert y[start] == symbol
        assert np.all(y[start + 1 : end + 1] == lt.BLANK)
        assert 1 <= end - start + 1 <= (4 if symbol != space else 3)
    frames = lt.labels(stream, "frames")
    assert lt.text_of(lt.collapse(frames), letters) == stream["text"]
    late = lt.labels(stream, "onset", delay=1)
    assert lt.text_of(lt.said(late), letters) == stream["text"]
    even = lt.labels(stream, "even")
    assert lt.text_of(lt.said(even), letters) == stream["text"]
    assert np.all(np.diff(lt.due(stream, even=True)) >= 1)


def test_readings_count_holds_inserts_and_misses() -> None:
    lexicon = lt.make_lexicon(0)
    stream = lt.make_stream(lexicon, 20, np.random.default_rng(2), pool="all")
    letters = 16
    perfect = lt.score(lt.labels(stream, "onset"), stream, letters)
    assert perfect["collapsed"]["characters"] == 1.0
    assert perfect["collapsed"]["words"] == 1.0
    assert perfect["held_steps"]["letters_once"] == 1.0
    held = lt.score(lt.labels(stream, "frames"), stream, letters)
    assert held["collapsed"]["characters"] == 1.0  # the collapse reads a held letter once
    assert held["held_steps"]["letters_mean"] > 1.5
    assert held["said"]["insertions"] > 0.3  # said on every frame: said more than once
    silent = lt.score(np.zeros(len(stream["track"]), dtype=np.int64), stream, letters)
    assert silent["collapsed"]["characters"] == 0.0
    assert silent["held_steps"]["letters_missed"] == 1.0


def test_edits_align_by_kind() -> None:
    out = lt.edits(list("short"), list("shorrk"))
    assert out["errors"] == 2
    assert out["insertions"] + out["substitutions"] + out["deletions"] == 2
    assert lt.reading("ab cd", "ab cd")["exact"]
    r = lt.reading("ab cx", "ab cd")
    assert r["words"] == 0.5 and r["word_matched"] == [True, False]


def test_probe_reads_a_linear_code() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(3, size=300)
    x = np.eye(3)[y] + rng.normal(0, 0.1, (300, 3))
    assert lt.probe(x, y) > 0.95


def test_a_short_life_runs_and_reads(tmp_path: Path) -> None:
    protocol = lt.load_protocol(PROTOCOL)
    sizes = {"training_words": 6, "probe_every": 3, "evaluation_words": 4}
    record = lt.brain_life("default", 0, protocol, tmp_path, sizes)
    assert record["parameters"] > 0
    assert [p["words"] for p in record["probes"]] == [3, 6]
    final = record["probes"][-1]
    assert 0.0 <= final["score"]["collapsed"]["characters"] <= 1.0
    assert "association" in final["probes"]


def test_forced_path_is_monotonic_and_follows_the_beliefs() -> None:
    scores = np.log(np.array([[.9, .05, .05], [.8, .1, .1], [.1, .8, .1], [.1, .1, .8],
                              [.1, .1, .8]]))
    assert lt.forced_path(scores).tolist() == [0, 0, 1, 2, 2]
    flat = np.zeros((3, 3))
    assert lt.forced_path(flat).tolist() == [0, 1, 2]  # each symbol takes a frame


def test_forced_labels_recover_the_heard_letters_from_sharp_beliefs() -> None:
    lexicon = lt.make_lexicon(0)
    stream = lt.make_stream(lexicon, 12, np.random.default_rng(3))
    y = lt.labels(stream, "even_frames")
    words = lt.words_of(stream)
    assert len(words) == 12
    assert lt.text_of(lt.collapse(y), 16) == stream["text"]
    truth = lt.labels(stream, "frames")
    beliefs = np.log(np.full((len(y), 18), 0.01))
    beliefs[np.arange(len(y)), truth] = 0.0
    right = lt.forced_labels(stream, beliefs, range(len(words)), y)
    assert right == 1.0
    letter_frames = stream["track"] >= 0
    np.testing.assert_array_equal(y[letter_frames], truth[letter_frames])
