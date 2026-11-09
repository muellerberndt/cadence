"""Regression gates for the public opt-in single-head attention recipe."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "contextual_choice_tests", Path(__file__).with_name("contextual_choice.py")
)
assert SPEC is not None and SPEC.loader is not None
subject = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = subject
SPEC.loader.exec_module(subject)


def test_final_question_is_literal_and_context_free() -> None:
    first = subject.observation("ask")
    second = subject.observation("ask")
    assert np.array_equal(first, second)
    assert first.shape == (1, subject.INPUTS)
    assert np.count_nonzero(first) == 2


def test_constructor_uses_the_public_opt_in_without_an_episodic_vote() -> None:
    brain = subject.compose(101, solve=64, module=8)
    assert brain.attention is True
    assert brain.recall is not None and brain.recall.window == 2
    assert brain.hippocampus is None and brain._episodic is None  # noqa: SLF001
    assert brain.memory_choice is False
    assert subject.RANDOM_BASELINE == 0.5


def test_cold_policy_preserves_only_the_recall_window_context() -> None:
    brain = subject.compose(102, solve=256, module=8)
    brain.act(subject.observation("context", 0), greedy=True)
    before = brain.recall.past.copy()
    subject.cold_policy(brain)
    assert np.array_equal(brain.recall.past, before)
    assert not brain.recall.previous.any()
    assert brain.basal_ganglia.state is None
    assert not brain.working_memory.trace.any()
    assert brain.learned_code.last is None


def test_retained_four_founder_receipt_is_source_bound_and_recomputed() -> None:
    path = Path(__file__).with_name("results") / "contextual-choice-v0870.json"
    ok, message = subject.verify_historical(path)
    assert ok, message
    receipt = subject.cd.Receipt.read(path)
    assert [founder["seed"] for founder in receipt.body["founders"]] == [0, 1, 2, 3]
    assert receipt.body["passed"]


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_multi_founder_action_relevant_selection_gate(seed: int) -> None:
    result = subject.run_founder(seed, trials=2)
    assert result["uniform_random_baseline_accuracy"] == 0.5
    assert result["passed"], result["gates"]
