"""The bandit of cadence#210: the lag, a short life of each arm, the summary, and the rule
acting only where an outcome is zero."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load():
    spec = importlib.util.spec_from_file_location("extinction_bandit",
                                                  Path(__file__).with_name("bandit.py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bd = _load()


def test_the_lag_is_the_first_run_of_twenty_with_eighteen_hits():
    assert bd.lag([0] * 5 + [1] * 20) == 3  # a window may hold two misses
    assert bd.lag([1, 0] * 10 + [1] * 18) == 16  # the window from 16 holds the misses 17, 19
    assert bd.lag([0, 1] * 15) is None


@pytest.mark.parametrize("theta", [0.0, 0.5])
def test_a_short_life_reads_every_phase(theta):
    row = bd.life({"seed": 3, "theta": theta, "fail": 0.0, "phase": 24, "phases": 3})
    assert [p["target"] for p in row["phases"]] == [0, 1, 0] and row["phase"] == 24
    for p in row["phases"]:
        assert 0.0 <= p["late"] <= 1.0 and len(p["record"]) == 2
    assert row["phases"][0]["old_arm_aroused_failures"] == 0  # nothing was paid before


def test_the_rule_changes_nothing_where_no_outcome_is_zero():
    lives = [bd.life({"seed": 3, "theta": t, "fail": -1.0, "phase": 24, "phases": 2})
             for t in (0.0, 0.5)]
    assert lives[0]["phases"] == lives[1]["phases"]


def test_the_summary_counts_the_lives_acquiring_each_later_phase():
    rows = [bd.life({"seed": s, "theta": 0.5, "fail": 0.0, "phase": 24, "phases": 2})
            for s in (1, 2)]
    (cell,) = bd.summarize(rows)
    assert cell["seeds"] == [1, 2] and set(cell) >= {"phase_1", "first_arm_record"}
    assert 0 <= cell["phase_1"]["acquired"] <= 2
    assert bd._seeds(["0-2", "7"]) == [0, 1, 2, 7]
