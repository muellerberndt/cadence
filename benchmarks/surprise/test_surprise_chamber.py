"""The surprise chamber (cadence#337) in short: both measures are read at every moment, and
the readings keep their shape."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _load():
    key = "surprise_chamber"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, HERE / "chamber.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


ch = _load()


def test_the_reader_reads_both_measures_of_one_brain_at_every_moment():
    reader = ch.Reader(ch.brain(0))
    rng = np.random.default_rng(0)
    first, second = ch.distinct(rng, 2, ch.STREAM)
    read = reader.moment(first)
    assert set(read) >= {"residual", "move", "residual_senses", "move_senses"}
    assert read["residual"] == read["move"] == 0.0  # from rest: nothing was carried in
    assert reader.brain.surprise.moments == reader.moved.moments == 1
    read = reader.moment(second)
    assert read["residual_senses"] > 2.9 and read["move_senses"] > 0.0
    assert read["residual_units"] > 0 and read["move_units"] > 0
    assert reader.brain.surprise.settings() == reader.moved.settings()  # one law, one set of genes


def test_an_input_held_reads_nothing_and_a_change_habituates_in_both_measures():
    found = ch.onset(0, held_moments=6, repeats=3, gap=2)
    for name in ch.MEASURES:
        assert found["held"][name] == 0.0, name
    first, third = found["changes"][0], found["changes"][2]
    assert first["residual_senses"] > 2.9 and first["move_senses"] > 0.0
    assert third["residual_senses"] == third["move_senses"] == 0.0
    assert found["new"]["residual_senses"] > 2.9 and found["new"]["move_senses"] > 0.0
    # under greedy acts no drive enters the association and motor regions: they carry no
    # residual at the start of a settle, and they do move
    moments = [found["held"], *found["changes"], found["return"], found["new"]]
    assert max(read["residual_inner"] for read in moments) == 0.0
    assert max(read["move_inner"] for read in moments) > 0.0


def test_a_deviant_on_spare_inputs_reads_above_the_standards_in_both_measures():
    found = ch.deviants(0, "repeat", "units", moments=60)
    assert found["deviants"] > 0
    for name in ("residual", "move"):
        assert found[name]["deviant"] > found[name]["standard"], name
        assert 0.5 <= found[name]["area"] <= 1.0, name
    assert ch.area([1.0, 2.0], [0.0, 0.0]) == 1.0 and ch.area([0.0], [0.0]) == 0.5
    assert np.isnan(ch.area([], [1.0]))


def test_a_lived_sequence_reads_nothing_and_an_element_on_spare_inputs_surprises():
    found = ch.learned(0, cycles=8)
    for arm in ("exposed", "taught"):
        rows = found[arm]
        assert set(rows) == {"first", "next_right", "regular", "swap", "repeat", "units",
                             "combination"}
        for name in ("residual", "move"):
            assert rows["first"][name]["share"] == 1.0, (arm, name)
            assert rows["units"][name]["share"] == 1.0, (arm, name)
    for name in ("residual", "move"):
        assert found["exposed"]["regular"][name]["share"] == 0.0, name


def test_the_cost_names_the_signal_s_parts():
    found = ch.cost(0, moments=5, names=("composed",))
    (row,) = found.values()
    assert row["neurons"] > 0 and row["synapses"] > 0
    for name in ("off", "on", "residual_us", "law_us", "move_us"):
        assert row[name] > 0.0, name


def test_the_signal_changes_nothing_else_to_the_bit():
    same = ch.bits()
    assert same["off equals unnamed"] and same["on equals unnamed"]
    assert len(same["unnamed"]) == 64
