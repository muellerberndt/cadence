"""Contracts of the encyclopedia chamber (cadence#311) on a tiny corpus: the corpus's table, the
instrument's moments and draws, and one small life of each arm, with no download."""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import chamber as ch  # noqa: E402
import corpus as cp  # noqa: E402
import facts as fx  # noqa: E402
from test_controls import tiny  # noqa: E402


def test_the_corpus_folds_names_drops_shared_ones_and_keeps_answers(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    head = "?s\t?sl\t?n\t?o\t?ol\n"

    def row(s: int, name: str, n: int, o: int, label: str) -> str:
        return (f"<http://www.wikidata.org/entity/Q{s}>\t\"{name}\"@en\t{n}\t"
                f"<http://www.wikidata.org/entity/Q{o}>\t\"{label}\"@en\n")

    people = [row(10 + i, f"Zoë Person{i}", 9 - i % 3, 1, "Painter") for i in range(6)]
    people += [row(30, "Same Name", 5, 1, "Painter"), row(31, "Same Name", 5, 1, "Painter")]
    people += [row(10, "Zoë Person0", 9, 2, "Rare Job")]  # a second occupation, named once
    for relation in cp.RELATIONS:
        body = head + ("".join(people) if relation == "occupation" else "")
        (raw / f"{relation}.tsv.gz").write_bytes(gzip.compress(body.encode()))
    out = tmp_path / "facts.tsv"
    cp.build(raw, out)
    lines = out.read_text().splitlines()
    assert lines[0].split("\t") == ["subject", "name", "articles", "relation", "object", "answer"]
    rows = [line.split("\t") for line in lines[1:]]
    assert len(rows) == 6  # the shared name is dropped with both its subjects
    assert {r[5] for r in rows} == {"painter"}  # the rare object is no answer
    assert rows[0][1] == "zoe person0" and int(rows[0][2]) >= int(rows[-1][2])
    summary = json.loads(out.with_suffix(".json").read_text())
    assert summary["facts"] == 6 and summary["answers"] == 1


def test_a_word_is_heard_as_its_form_and_a_fact_marks_its_turn(tmp_path: Path) -> None:
    corpus = fx.Corpus(tiny(tmp_path))
    lex = corpus.lexicon
    assert lex.inputs == fx.WIDTH + 1 and lex.actions == len(set(lex.answers))
    for word in ("arlo", "capital", "?"):
        x = lex.moment(word)
        assert x.sum() == fx.ON and not x[fx.WIDTH]
        assert np.array_equal(x, lex.moment(word))
    assert not np.array_equal(lex.moment("arlo"), lex.moment("bevin"))
    fact = corpus.reading[0]
    told = lex.moments(fx.tell(fact))
    assert all(label is None for _, label in told)
    at = [i for i, (x, _) in enumerate(told) if x[fx.WIDTH]]
    assert len(at) == 1 and np.array_equal(told[at[0] + 1][0], lex.moment(fact.answer))
    asked = lex.moments(fx.ask(fact, "held", 0))
    assert asked[-2][0][fx.WIDTH] == 1 and not asked[-1][0].any()
    assert asked[-1][1] == lex.action[fact.answer]


def test_lessons_and_the_reading_share_no_subject_and_chains_need_both_facts(
        tmp_path: Path) -> None:
    corpus = fx.Corpus(tiny(tmp_path))
    assert not {f.subject for f in corpus.lessons} & {f.subject for f in corpus.reading}
    for first, second in fx.chains(corpus.reading):
        assert first.answer == second.name[0] and (first.relation, second.relation) in fx.CHAINS
    rng = np.random.default_rng(0)
    session = fx.lesson_session(corpus, rng, 4, two=0.5)
    asked = [t for t in session if t.asked]
    assert len(asked) == 4 and all(not t.asked for t in session[: len(session) - 4])
    told = {" ".join(t.said) for t in session if not t.asked}
    lesson_names = {" ".join(f.name) for f in corpus.lessons}
    assert all(any(n in s for n in lesson_names) for s in told)


def test_a_small_life_reads_its_own_records(tmp_path: Path) -> None:
    spec = {"data": str(tiny(tmp_path)), "founder": 0, "size": 40, "sessions": 3, "facts": 3,
            "questions": 6, "small": 4, "module": 16, "slack": 200}
    rows = {arm: ch.run_cell({**spec, "arm": arm})[0] for arm in ch.ARMS}
    best = rows["records"]
    assert best["records"] == 40 + 4  # the corpus and the revisions, each told once
    scores = best["scores"]
    assert scores["found trained"]["right"] == 1.0 and scores["found held"]["right"] == 1.0
    assert scores["found revised"]["right"] == 1.0
    assert scores.get("found erased", {"right": 0.0})["right"] == 0.0
    for group, n in (("trained", 6), ("held", 6), ("synonym", 4), ("revised", 4),
                     ("erased", 4), ("stale", 4)):
        assert scores[group]["questions"] == n
    assert best["intake_seconds_per_fact"] > 0 and best["answer_macs"] > 0
    assert rows["previous_word"]["records"] is None
    assert rows["previous_word"]["parameters"] < best["parameters"]
