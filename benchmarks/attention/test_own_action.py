"""Regression gates for the executed command in the recall memory (``recall_action``)."""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np

SPEC = importlib.util.spec_from_file_location(
    "own_action_tests", Path(__file__).with_name("own_action.py")
)
assert SPEC is not None and SPEC.loader is not None
subject = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = subject
SPEC.loader.exec_module(subject)

RECEIPT = Path(__file__).with_name("results") / "own-action.json"


def test_the_answers_follow_the_brains_own_history() -> None:
    assert [subject.label("alternate", h) for h in ([], [0], [0, 1])] == [0, 1, 0]
    assert [subject.label("timing", h) for h in ([], [0], [0, 0], [0, 0, 1])] == [0, 0, 1, 0]
    # a constant policy scores near zero: its own history makes every answer the other action
    for task in subject.TASKS:
        history: list[int] = []
        hits = 0
        for _ in range(subject.LENGTH):
            hits += subject.label(task, history) == 0
            history.append(0)
        assert hits <= 1 + subject.CADENCE


def test_erasing_removes_every_command_and_nothing_else() -> None:
    brain = subject.compose(0, "action")
    for _ in range(3):
        brain.act(subject.X, greedy=True)
    events = brain.recall.values[..., : -brain.recall.command_width].copy()
    subject.erase(brain)
    recall = brain.recall
    assert not recall.values[..., -recall.command_width:].any() and not recall.acted.any()
    assert not recall.keys[..., -recall.command_width:].any()
    np.testing.assert_array_equal(recall.values[..., : -recall.command_width], events)


def test_retained_receipt_is_source_bound_and_recomputed() -> None:
    ok, message = subject.verify_historical(RECEIPT)
    assert ok, message
    body = subject.cd.Receipt.read(RECEIPT).body
    assert body["passed"] and all(body["gates"].values())
    assert sorted({r["seed"] for r in body["runs"]}) == list(subject.SEEDS)


def test_the_check_refuses_a_decision_its_readings_do_not_support() -> None:
    body = copy.deepcopy(subject.cd.Receipt.read(RECEIPT).body)
    run = next(r for r in body["runs"] if r["arm"] == "action")
    run["score"] = 0.0
    assert subject.historical_check(body) is not None
