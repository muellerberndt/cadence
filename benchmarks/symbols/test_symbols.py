"""The symbols chamber's world and readings, small."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import chamber as ch  # noqa: E402
from world import KINDS, WORDS, a_thing  # noqa: E402


def test_the_trainer_hears_every_take_as_its_word():
    things, ear, words = ch.setting()
    rng = np.random.default_rng(0)
    assert all(words.heard(words.take(word, rng)) == word for word in WORDS for _ in range(3))
    assert words.heard(np.zeros((1, ear.width))) is None
    assert things.width == 56 and ear.width == 640


def test_a_named_thing_comes_back_from_its_name_and_not_without_records():
    things, ear, words = ch.setting()
    rng = np.random.default_rng(1)
    brain = ch.compose(0, things, ear)
    ch.listen(brain, words, rng, reps=1)
    ch.naming(brain, things, words, rng, rounds=2)
    back = brain.complete({"ear": words.take("banana", rng)})["eyes"]
    assert ch.kind_of(back, things) == "banana"
    brain.bound.store.values[:] = 0.0
    assert ch.kind_of(brain.complete({"ear": words.take("banana", rng)})["eyes"], things) is None


def test_a_reward_teaches_a_name_and_the_controls_run():
    things, ear, words = ch.setting()
    rng = np.random.default_rng(2)
    brain = ch.compose(0, things, ear)
    ch.listen(brain, words, rng, reps=1)
    items = [(a_thing(rng, "cup", "white"), None, "cup")]
    paid = ch.exchanges(brain, things, words, rng, items, paid=True)
    assert paid["exchanges"][0] is not None and paid["exchanges"][0] <= ch.BUDGET
    assert ch.table_control(items, rng)["exchanges"][0] is not None
    assert len(ch.delta_control(brain, things, words, items, rng)["exchanges"]) == 1
    assert set(KINDS) >= {"cup", "apple", "banana", "key"}


def test_gate_status_requires_one_complete_declared_panel(tmp_path, capsys):
    import importlib.util
    import json

    import pytest

    folder = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("symbols_summary", folder / "summary.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    row = json.loads((folder / "results" / "cortex_reads.json").read_text())[0]
    receipt = tmp_path / "panel.json"

    def verdict(founders, arm="code"):
        receipt.write_text(json.dumps([dict(row, founder=f, cortex_reads=arm) for f in founders]))
        module.main([str(receipt)])
        return capsys.readouterr().out

    with pytest.raises(SystemExit, match="duplicate"):
        verdict([1010400] * 10)
    assert "S1: 0 of 0 -> incomplete" in verdict([])
    assert "S1: 10 of 10 -> incomplete" in verdict(range(1010400, 1010410))
    assert "S1: 12 of 12 -> incomplete" in verdict(range(1010400, 1010412), "both")
    assert "S1: 12 of 12 -> holds" in verdict(range(1010400, 1010412))
    for counts in ([], [0, 12, 14, 16], [12, 14, 16], [12, 14, 16, 81]):
        row["Y3"]["names_paid"]["exchanges"] = counts
        assert not module.gates(row)["S4"]
