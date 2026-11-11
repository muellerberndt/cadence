"""Regression gates for looking back by position (``recall_position``)."""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np

SPEC = importlib.util.spec_from_file_location(
    "look_back_tests", Path(__file__).with_name("look_back.py")
)
assert SPEC is not None and SPEC.loader is not None
subject = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = subject
SPEC.loader.exec_module(subject)

RECEIPT = Path(__file__).with_name("results") / "look-back.json"


def test_the_answer_depends_only_on_the_cued_place() -> None:
    rng = np.random.default_rng(0)
    for _ in range(64):
        symbols, cue, label = subject.episode(rng)
        assert len(symbols) == subject.LENGTH
        assert label == (symbols[0] if cue == subject.FIRST else symbols[-1])


def test_the_arms_differ_only_in_the_position_term() -> None:
    with_term, without = subject.compose(0, 32), subject.compose(0, 0)
    assert with_term.recall.position == 32 and with_term.recall.position_maps
    assert without.recall.position == 0 and not without.recall.position_maps
    assert with_term.recall.recency == without.recall.recency == 0.0
    assert with_term.recall.boundary and without.recall.boundary
    assert with_term.hippocampus is None and with_term._episodic is None  # noqa: SLF001
    assert with_term.describe()["layout"] == without.describe()["layout"]


def test_retained_receipt_is_source_bound_and_recomputed() -> None:
    ok, message = subject.verify_historical(RECEIPT)
    assert ok, message
    body = subject.cd.Receipt.read(RECEIPT).body
    assert sorted(f["seed"] for f in body["founders"]) == [0, 1, 2, 3]
    # with the boundary every founder's term answers both cues and beats the memory without it
    assert all(f["gates"]["beats_the_memory_without_it"] for f in body["founders"])
    assert [f["passed"] for f in body["founders"]] == [True, True, True, True]
    assert body["passed"]


def test_the_check_refuses_a_decision_its_readings_do_not_support() -> None:
    body = copy.deepcopy(subject.cd.Receipt.read(RECEIPT).body)
    body["founders"][0]["with_position"]["first"] = 0.5
    assert subject.historical_check(body) is not None
    body = copy.deepcopy(subject.cd.Receipt.read(RECEIPT).body)
    body["founders"] = body["founders"][:3]
    assert subject.historical_check(body) is not None
