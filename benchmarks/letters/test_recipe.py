"""The speech recipe (recipe.py) end to end on a tiny synthetic corpus, in CI time."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("letters_recipe", _here / "recipe.py")
assert _spec is not None and _spec.loader is not None
recipe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(recipe)


def test_the_recipe_trains_reads_adapts_and_streams(tmp_path, capsys):
    out = tmp_path / "report.json"
    report = recipe.main(["--synthetic", "--utterances", "16", "--held", "6", "--adapt", "4",
                          "--passes", "2", "--batch", "4", "--align-every", "2", "--fields", "4",
                          "--width", "16", "--out", str(out)])
    assert out.exists() and report["parameters"] > 0
    assert [r["pass"] for r in report["passes"]] == [1, 2]
    for r in report["passes"]:
        assert r["characters"] <= 1.0 and r["words"] <= 1.0
        assert r["training_seconds_per_audio_hour"] > 0
    adapt = report["adapt"]
    assert adapt["corrections"] >= 1
    assert set(adapt["characters_after"]) == {"speaker", "others"}
    stream = report["stream"]
    assert stream["step_median_s"] > 0 and isinstance(stream["said"], str)
    printed = capsys.readouterr().out
    assert '"characters"' in printed


def test_the_synthetic_corpus_and_the_reading():
    corpus = recipe.synthetic(6, seed=0)
    assert all(set(u["text"]) <= set("agnot ") for u in corpus)
    assert {u["speaker"] for u in corpus} == {"s0", "s1", "s2"}
    assert {u["speaker"] for u in recipe.synthetic(2, seed=1, speakers=1, first=3)} == {"s3"}
    assert recipe.edits(list("abc"), list("abd")) == 1
    assert recipe.score(["go no"], ["go no"]) == {"characters": 1.0, "words": 1.0}
    assert np.isclose(recipe.score(["go"], ["go no"])["words"], 0.5)
