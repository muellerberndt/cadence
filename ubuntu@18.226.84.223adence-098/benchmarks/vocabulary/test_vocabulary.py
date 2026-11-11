"""The scaled language is the grammar chamber at the chamber's size, and sound at any size."""

from __future__ import annotations

import copy
import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "grammar"))


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


gr = _load("grammar", HERE.parent / "grammar" / "grammar.py")
language = _load("language", HERE / "language.py")
life = _load("life", HERE / "life.py")


def same_row(lang, ours, theirs) -> None:
    assert ours.family == theirs.family
    assert ours.target == theirs.target
    assert ours.force == theirs.force
    assert len(ours.pre) == len(theirs.pre)
    for a, b in zip(ours.pre, theirs.pre, strict=True):
        assert np.array_equal(lang.dense([a])[0], b)


def test_the_chamber_at_its_own_size_is_the_chamber() -> None:
    lang = language.Language(4, 2)
    assert (lang.inputs, lang.actions) == (gr.INPUTS, gr.ACTIONS)
    assert lang.near == gr.NEAR and lang.end == gr.END and lang.opens == gr.OPENS
    assert lang.go == gr.C["GO"] and lang.speak == gr.C["SPEAK"]
    for founder in range(4):
        ours, theirs = lang.panels(founder), gr.panels(founder)
        assert set(ours) == set(theirs)
        for name in theirs:
            for a, b in zip(ours[name], theirs[name], strict=True):
                for x, y in zip(a, b, strict=True):
                    same_row(lang, x, y)
        s = gr.Split.of(founder)
        r1 = np.random.default_rng(founder)
        r2 = np.random.default_rng(founder)
        ours_split = lang.split(founder)
        assert ours_split.triples == s.triples and ours_split.pair == s.pair
        for _ in range(30):
            for x, y in zip(lang.train_batch(r1, ours_split), gr.train_batch(r2, s, "g1"),
                            strict=True):
                same_row(lang, x, y)


def test_the_life_at_the_chamber_size_is_the_chamber_life() -> None:
    warnings.simplefilter("ignore")
    lang = language.Language(4, 2)
    settings = life.ARMS["Fd"]
    gr.ARMS["vocab_fd"] = settings
    ours = life.make_brain(lang, "Fd", 3)
    theirs = gr.make_brain("vocab_fd", 3)
    assert ours.parameters() == theirs.parameters() == 30362
    split_ours, split_theirs = lang.split(3), gr.Split.of(3)
    r1 = np.random.default_rng(np.random.SeedSequence([3, 11]))
    r2 = np.random.default_rng(np.random.SeedSequence([3, 11]))
    for _ in range(6):
        said_ours = life.speak(lang, ours, lang.train_batch(r1, split_ours), teach=True)
        said_theirs = gr.speak(theirs, gr.train_batch(r2, split_theirs, "g1"), teach=True)
        assert said_ours == said_theirs
    rows_ours, rows_theirs = lang.panels(3)["path"][0], gr.panels(3)["path"][0]
    assert life.speak(lang, ours, rows_ours, teach=False) == gr.speak(
        theirs, rows_theirs, teach=False)


@pytest.mark.parametrize("words", [200, 1000, 5000])
def test_sizes_by_class(words: int) -> None:
    nouns, verbs = language.by_size(words)
    lang = language.Language(nouns, verbs)
    assert lang.words == words and lang.actions == words
    assert lang.inputs == words + nouns + verbs + 11


@pytest.mark.parametrize("words", [18, 200, 1000])
def test_every_panel_and_batch_is_answerable_from_what_is_heard(words: int) -> None:
    lang = language.Language(*language.by_size(words))
    tests = lang.panels(1, 2)
    for name in language.G1_PANELS + ("ctx_agent", "ctx_verb", "ctx_dyck", "g2", "g2_long"):
        for batch in tests[name]:
            for row in batch:
                assert lang.solve(row.pre) == row.target, name
    rng = np.random.default_rng(5)
    split = lang.split(1)
    for _ in range(20):
        for row in lang.train_batch(rng, split):
            assert lang.solve(row.pre) == row.target


def test_the_split_holds_its_shares_and_training_keeps_out() -> None:
    lang = language.Language(*language.by_size(1000))
    split = lang.split(0)
    rng = np.random.default_rng(0)
    pairs = rng.integers(lang.trained, size=(20000, 2))
    share = np.mean([split.held_pair(int(p), int(q)) for p, q in pairs if p != q])
    triples = np.mean([split.held_triple((int(rng.integers(lang.trained)),
                                          int(rng.integers(2))), int(rng.integers(lang.verbs)))
                       for _ in range(20000)])
    assert abs(share - 1 / 6) < 0.02 and abs(triples - 1 / 8) < 0.02
    for _ in range(200):
        row = lang.trained_clause(rng, split, 1, False)
        m = row.meta
        assert not split.held(tuple(m["agent"]), tuple(m["patient"]), m["verb"])
    for batch in lang.panels(0, 2)["clause_held"]:
        for row in batch:
            m = row.meta
            assert split.held(tuple(m["agent"]), tuple(m["patient"]), m["verb"])


def test_fresh_stems_stay_out_of_training_and_panels() -> None:
    lang = language.Language(65, 32, fresh=16)
    rng = np.random.default_rng(0)
    split = lang.split(0)
    fresh_words = {lang.noun(i, p) for i in range(lang.trained, lang.nouns) for p in (0, 1)}
    fresh_concepts = {lang.noun_concept0 + i for i in range(lang.trained, lang.nouns)}
    for _ in range(100):
        for row in lang.train_batch(rng, split):
            assert not fresh_words & set(row.target)
            assert not any(fresh_concepts & set(m) or fresh_words & set(m) for m in row.pre)
    for batches in lang.panels(0, 2).values():
        for batch in batches:
            for row in batch:
                assert not fresh_words & set(row.target)
                assert not any(fresh_concepts & set(m) or fresh_words & set(m) for m in row.pre)


def test_a_mention_is_a_path_the_solver_can_answer() -> None:
    hearing = _load("hearing", HERE / "hearing.py")
    lang = language.Language(65, 32, fresh=16)
    heard, other = hearing.plans(lang)
    assert {s for s, _ in heard} == {s for s, _ in other} == set(range(49, 65))
    assert all(a[1] != b[1] for a, b in zip(heard, other, strict=True))
    rng = np.random.default_rng(0)
    for row, (stem, _) in zip(hearing.mentioned(lang, rng, heard), heard, strict=True):
        assert lang.solve(row.pre) == row.target
        assert row.target[2] == lang.noun(stem, 0)
    for row in hearing.sentences(lang, rng, other):
        assert lang.solve(row.pre) == row.target


def test_heard_pieces_say_their_word_and_novel_stems_have_taught_pieces(tmp_path: Path) -> None:
    pieces = _load("pieces", HERE / "pieces.py")
    words = [f"w{a}{b}" for a in "abcdefgh" for b in "ijklmnop"]
    split = {w: [w[:2], "##" + w[2]] for w in words}
    names = sorted({p for w in words for p in split[w]})
    lang = pieces.PieceLanguage({"words": words, "pieces": names, "split": split}, swap=True)
    s = lang.split(0)
    taught = {p for table in (s.nouns, s.verbs) for stems in table.values() for i in stems
              for p in (lang.noun_stems[i] if table is s.nouns else lang.verb_stems[i])}
    for table, stems in ((s.novel_nouns, lang.noun_stems), (s.novel_verbs, lang.verb_stems)):
        for bucket in table.values():
            for i in bucket:
                assert set(stems[i]) <= taught
    rng = np.random.default_rng(0)
    for _ in range(40):
        batch = lang.train_batch(rng, s)
        assert len({len(r.pre) for r in batch}) == 1
        assert len({len(r.target) for r in batch}) == 1
        for r in batch:
            for m in r.pre:
                assert all(0 <= c < lang.inputs for c in m)
            assert all(0 <= w < lang.actions for w in r.target)
            assert len(r.meta["sources"]) == len(r.target)
    for name, batches in lang.panels(0, 2).items():
        for batch in batches:
            assert len({len(r.pre) for r in batch}) == 1, name
            if name.startswith("novel") or name.startswith("swap_novel"):
                for r in batch:
                    held = [lang.noun_stems[stem] for stem, _ in (
                        [tuple(r.meta["agent"])] if "agent" in r.meta["novel"] else [])]
                    assert all(stem in [lang.noun_stems[i] for b in s.novel_nouns.values()
                                        for i in b] for stem in held)
    row = lang.panels(0, 2)["taught"][0][0]
    hit = lang.score(row, list(row.target))
    assert hit["exact"] and hit["agent_word"] and hit["patient_word"] and hit["agree"]


def test_the_verb_flag_adds_one_input_and_keeps_every_sentence() -> None:
    plain, flagged = language.Language(65, 32), language.Language(65, 32, verb_flag=True)
    assert flagged.inputs == plain.inputs + 1
    a, b = np.random.default_rng(3), np.random.default_rng(3)
    for _ in range(4):
        x, y = plain.clause(a, 1), flagged.clause(b, 1)
        assert x.target == y.target
        assert set(y.pre[0]) == {*x.pre[0], flagged.verb_flag}
        assert flagged.solve(y.pre) == y.target


def test_the_coded_maps_start_as_a_brain_without_them_and_learn() -> None:
    warnings.simplefilter("ignore")
    lang = language.Language(*language.by_size(200))
    plain = life.make_brain(lang, "Fd", 2)
    coded = life.make_brain(lang, "Fd", 2, {"map_code": 32})
    assert coded.parameters() < plain.parameters()
    rows = lang.panels(2, 1)["clause"][0]
    assert life.speak(lang, plain, rows, teach=False) == life.speak(lang, coded, rows, teach=False)
    w_before = coded.recall.head_maps[0][0].copy()
    life.speak(lang, coded, lang.train_batch(np.random.default_rng(1), lang.split(2)), teach=True)
    assert coded.recall.head_maps[0][0].shape == (32, lang.inputs)
    assert np.abs(coded.recall.head_maps[0][0] - w_before).max() > 0
    fork = __import__("copy").deepcopy(coded)
    assert life.speak(lang, fork, rows, teach=False) == life.speak(lang, coded, rows, teach=False)


def test_the_start_flag_marks_each_heard_first_piece_and_changes_no_sentence() -> None:
    pieces = _load("pieces", HERE / "pieces.py")
    words = [f"w{a}{b}" for a in "abcdefgh" for b in "ijklmnop"]
    split = {w: [w[:2], "##" + w[2]] for w in words}
    names = sorted({p for w in words for p in split[w]})
    lexicon = {"words": words, "pieces": names, "split": split}
    plain = pieces.PieceLanguage(lexicon, swap=True)
    flagged = pieces.PieceLanguage(lexicon, swap=True, start=True)
    assert flagged.inputs == plain.inputs + 1
    a, b = np.random.default_rng(1), np.random.default_rng(1)
    s = plain.split(0)
    for _ in range(30):
        for x, y in zip(plain.train_batch(a, s), flagged.train_batch(b, s), strict=True):
            assert x.target == y.target
            assert [set(m) - {flagged.start} for m in y.pre] == [set(m) for m in x.pre]
            if x.family == "clause":
                heard = [i for i, m in enumerate(y.pre) if flagged.start in m]
                assert len(heard) == 3 + x.meta["L"]  # one first piece a word


def test_made_up_stems_come_from_taught_pieces_and_leave_the_panels_of_no_share_alone() -> None:
    pieces = _load("pieces", HERE / "pieces.py")
    words = [f"w{a}{b}" for a in "abcdefgh" for b in "ijklmnop"]
    split = {w: [w[:2], "##" + w[2]] for w in words}
    names = sorted({p for w in words for p in split[w]})
    lexicon = {"words": words, "pieces": names, "split": split}
    plain = pieces.PieceLanguage(lexicon, swap=True)
    made = pieces.PieceLanguage(lexicon, swap=True, pseudo=0.5)
    s = plain.split(0)
    assert made.split(0) == s
    assert set(s.firsts) and set(s.laters)
    for name, batches in plain.panels(0, 2).items():
        for x_batch, y_batch in zip(batches, made.panels(0, 2)[name], strict=True):
            for x, y in zip(x_batch, y_batch, strict=True):
                assert x.target == y.target, name
    rng = np.random.default_rng(0)
    found = False
    for _ in range(60):
        for row in made.train_batch(rng, s):
            assert all(0 <= w < made.actions for w in row.target)
            assert len(row.meta["sources"]) == len(row.target)
            if row.family == "clause":
                stem = row.meta["agent"][0]
                if not isinstance(stem, int):
                    found = True
                    assert stem[0] in s.firsts and all(p in s.laters for p in stem[1:])
    assert found
    for batch in made.panels(0, 2)["pseudo"]:
        assert len({len(r.pre) for r in batch}) == 1
        for r in batch:
            assert not isinstance(r.meta["agent"][0], int)


def test_a_growing_vocabulary_draws_from_what_it_holds_and_the_panels_stay_whole() -> None:
    lang = language.Language(*language.by_size(1000))
    tests = lang.panels(0, 1)
    split = lang.split(0)
    lang.grown(65, 32)
    rng = np.random.default_rng(0)
    for _ in range(40):
        for row in lang.train_batch(rng, split):
            if row.family == "clause":
                assert row.meta["agent"][0] < 65 and row.meta["patient"][0] < 65
                assert row.meta["verb"] < 32
            assert lang.solve(row.pre) == row.target
            if row.family == "path":
                assert all(lang.is_noun(w) and w % lang.nouns < 65 for w in row.meta["chain"])
    assert any(r.meta["agent"][0] >= 65 for r in tests["clause"][0])
    lang.grown(10**6, 10**6)
    assert lang.trained == lang.nouns and lang.spoken_verbs == lang.verbs


@pytest.mark.parametrize("extra", [{}, {"full": [1]},
                                   {"grow": {"words": 200, "hold": 0, "ramp": 1}}])
def test_a_life_reads_at_its_checkpoints(extra: dict, tmp_path: Path) -> None:
    out = tmp_path / "life.jsonl"
    rows = life.run_life({"words": 1000 if "grow" in extra else 200, "founder": 0,
                          "checkpoints": [1], "locate": False, "panel_batches": 1,
                          "out": str(out), **extra})
    assert rows[0]["batches"] == 1 and "clause" in rows[0]["panels"]
    assert out.read_text().count("\n") == 1


def _taught_small(settings: dict | None = None, batches: int = 3) -> tuple:
    warnings.simplefilter("ignore")
    lang = language.Language(4, 2)
    brain = life.make_brain(lang, "Fd", 1, settings)
    rng = np.random.default_rng(np.random.SeedSequence([1, 11]))
    split = lang.split(1)
    for _ in range(batches):
        life.speak(lang, brain, lang.train_batch(rng, split), teach=True)
    return lang, brain


def test_the_terms_sum_to_the_attention_the_read_uses() -> None:
    lang, brain = _taught_small()
    tests = lang.panels(1, 1)
    out = life.terms(lang, brain, tests["clause"])
    assert out["check"] < 1e-6
    whole = out["all"]
    assert whole["n"] > 0 and 0.0 <= whole["missed"] <= 1.0
    parts = sum(whole["gap"][name] for name in life.TERMS)
    assert abs(parts - whole["gap"]["total"]) < 0.01


def test_the_cortex_emulation_reads_unit_norm_and_a_fork_keeps_it() -> None:
    warnings.simplefilter("ignore")
    lang = language.Language(4, 2)
    plain = life.make_brain(lang, "Fd", 1)
    unit = copy.deepcopy(life.make_brain(lang, "Fd", 1, {"cortex": {"unit": True}}))
    assert type(unit).__name__ == "Brain" and type(unit) is not type(plain)
    x = lang.dense([(lang.go,)] * 2)
    for brain in (plain, unit):
        brain.reset()
        brain.act(x, greedy=True)
    raw = np.asarray(plain.recall.previous)
    assert np.linalg.norm(raw, axis=1).min() > 0
    plain.recall.previous = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    for brain in (plain, unit):
        brain.act(x, greedy=True)
    np.testing.assert_allclose(plain.recall.attended_heads[0], unit.recall.attended_heads[0])


def test_a_cortex_rate_of_zero_keeps_every_maps_cortex_columns_at_birth() -> None:
    lang, brain = _taught_small({"cortex": {"rate": 0.0}}, batches=2)
    assert brain.recall.head_map(0)[0].any()
    for name in life.CORTEX_MAPS:
        for head in getattr(brain.recall, name, None) or []:
            assert not np.asarray(head[1]).any()
