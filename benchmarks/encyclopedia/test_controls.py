"""Contracts of the encyclopedia controls (cadence#311) on a tiny corpus: the moments they hear,
the BM25 scores they read, and one small cell of each model."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import facts as fx  # noqa: E402

COUNTRIES = ("arlo", "bevin", "corra", "dunmar", "estel", "farrow", "gilde", "harn")
CITIES = ("ashby", "brill", "cobb", "dale", "eske", "fenn", "gorm", "hythe")
JOBS = ("painter", "singer", "farmer")


def tiny(folder: Path) -> Path:
    path = folder / "facts.tsv"
    rows = []
    for i, (country, city) in enumerate(zip(COUNTRIES, CITIES, strict=True)):
        rows.append((f"Q{100 + i}", country, "9", "capital", f"Q{200 + i}", city))
        rows.append((f"Q{100 + i}", country, "9", "continent", "Q46",
                     "europe" if i % 2 else "asia"))
    for i in range(48):
        name = f"person{i} smith"
        rows.append((f"Q{300 + i}", name, "3", "citizenship", f"Q{100 + i % 8}",
                     COUNTRIES[(3 * i) % 8]))
        rows.append((f"Q{300 + i}", name, "3", "occupation", "Q1", JOBS[i % 3]))
    with path.open("w", newline="") as sink:
        out = csv.writer(sink, delimiter="\t")
        out.writerow(("subject", "name", "articles", "relation", "object", "answer"))
        out.writerows(rows)
    return path


def test_the_moments_are_the_lexicon_s(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    import controls as co

    corpus = fx.Corpus(tiny(tmp_path))
    assert corpus.lessons and corpus.reading
    mo = co.Moments(corpus.lexicon)
    fact = corpus.reading[0]
    for turn in (fx.tell(fact), fx.ask(fact, "held", 1), fx.ask_two(fact, fact)):
        x, y = mo.batch([mo.turn(turn)])
        want = corpus.lexicon.moments(turn)
        assert np.array_equal(x[0], np.stack([m for m, _ in want]))
        assert [int(v) for v in y[0]] == [co.IGNORE if a is None else a for _, a in want]
    ids, labels = mo.turn(fx.tell(fact), told=True)
    asked = corpus.lexicon.moments(fx.Turn(fx.tell(fact).said[:-2], fact.answer))
    assert len(ids) == len(asked) and labels[-1] == corpus.lexicon.action[fact.answer]
    assert ids[-2:] == [co.TURN, co.SILENT]


def test_the_index_reads_bm25_okapi_s_scores(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    rank_bm25 = pytest.importorskip("rank_bm25")
    import controls as co

    corpus = fx.Corpus(tiny(tmp_path))
    sentences = [fx.tell(f).said for f in corpus.reading]
    index = co.Index(sentences)
    bm25 = rank_bm25.BM25Okapi(sentences)
    for f in corpus.reading[:5]:
        query = fx.ask(f, "trained", 0).said
        assert np.allclose(index.scores(query)[0], bm25.get_scores(query))
        hits, _ = index.top(query, 3)
        assert corpus.reading.index(f) in hits


@pytest.mark.parametrize("model", ["closed_lstm", "closed_transformer", "context_transformer",
                                   "retrieval"])
def test_a_small_cell_of_each_model_reads_the_chamber_s_groups(tmp_path: Path,
                                                               model: str) -> None:
    pytest.importorskip("torch")
    if model == "retrieval":
        pytest.importorskip("rank_bm25")
    import controls as co

    spec = {"data": str(tiny(tmp_path)), "founder": 0, "model": model, "size": 40,
            "target": 20000, "sessions": 4, "facts": 3, "questions": 5, "small": 3,
            "window": 64, "k": 2, "epochs": 2, "budget": 0.05, "revision_budget": 0.05,
            "threads": 1}
    (row,) = co.run_cell(spec)
    assert abs(row["parameters"] - 20000) / 20000 < 0.25
    scores = row["scores"]
    for group, n in (("trained", 5), ("held", 5), ("synonym", 3), ("revised", 3),
                     ("erased", 3), ("stale", 3)):
        assert scores[group]["questions"] == n
    if model == "context_transformer":
        assert scores["in window erased"]["right"] == 0.0
        assert row["window_facts"] > 0
    if model == "retrieval":
        assert scores["found trained"]["right"] > 0.5
    assert row["answer_ms"] > 0 and row["answer_macs"] > 0
    assert row["memory_bytes"] >= 4 * row["parameters"]
    plan = {"founders": [0], "models": ["retrieval", {"model": "closed_lstm", "label": "x"}],
            "sizes": [10, 20], "window": 64}
    todo = co.cells(plan)
    assert len(todo) == 4 and {c["window"] for c in todo} == {64}
