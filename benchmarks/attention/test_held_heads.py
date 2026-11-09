"""Regression gates for several facts in one settle with held heads (``recall_heads``)."""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np

SPEC = importlib.util.spec_from_file_location(
    "held_heads_tests", Path(__file__).with_name("held_heads.py")
)
assert SPEC is not None and SPEC.loader is not None
subject = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = subject
SPEC.loader.exec_module(subject)

RECEIPT = Path(__file__).with_name("results") / "held-heads.json"


def test_held_out_pairs_are_both_orders_and_never_taught() -> None:
    for rule in ("report", "flee"):
        task = subject.Task(rule)
        assert all((b, a) in task.held_out for a, b in task.held_out)
        rng = np.random.default_rng(0)
        assert not any(p[:2] in task.held_out for p in task.sample(rng, 64, held_out=False))
        assert all(p[:2] in task.held_out for p in task.sample(rng, 16, held_out=True))


def test_the_heads_arm_is_the_public_recipe() -> None:
    task = subject.Task("report")
    two, one = task.compose(0, 2), task.compose(0, 1)
    assert two.attention and two.recall.heads == 2 and one.recall.heads == 1
    assert (two.recall.address, two.recall.temperature) == (0.01, 0.25)
    assert (one.recall.address, one.recall.temperature) == (0.01, 0.25)
    assert two.hippocampus is None and two._episodic is None  # noqa: SLF001


def test_the_flee_answer_needs_both_facts() -> None:
    task = subject.Task("flee")
    even = [task.labels([(a, b, 0, 0)])[0] == a for a in range(10) for b in range(0, 10, 2)]
    odd = [task.labels([(a, b, 0, 0)])[0] == a for a in range(10) for b in range(1, 10, 2)]
    assert all(even)  # with the predator at an even position the food's position answers
    assert sum(odd) == 5  # else the predator decides, the food's position only by coincidence


def test_retained_receipt_is_source_bound_and_recomputed() -> None:
    ok, message = subject.verify_historical(RECEIPT)
    assert ok, message
    body = subject.cd.Receipt.read(RECEIPT).body
    assert body["passed"]
    assert sorted((f["rule"], f["seed"]) for f in body["founders"]) == sorted(
        (rule, seed) for rule in ("report", "flee") for seed in (0, 1, 2, 3)
    )


def test_the_check_refuses_a_decision_its_readings_do_not_support() -> None:
    body = copy.deepcopy(subject.cd.Receipt.read(RECEIPT).body)
    assert subject.historical_check(body) is None
    body["founders"][0]["arms"]["heads_2"]["trained"]["accuracy"] = 0.5
    assert subject.historical_check(body) is not None


def test_a_short_life_runs_the_whole_protocol() -> None:
    result = subject.run_founder(0, "report", lessons=4)
    assert set(result["arms"]) == {"heads_1", "heads_2"}
    assert set(result["gates"]) == {
        "training_completed", "trained_accuracy", "beats_one_head", "beats_uniform_random",
        "distinct_heads",
    }
    assert result["arms"]["heads_2"]["training"]["accepted"] == 4
