"""Answers on tokens no lesson used (benchmarks/sequence/held.py)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("held_tokens", HERE / "held.py")
assert spec is not None and spec.loader is not None
held = importlib.util.module_from_spec(spec)
sys.modules["held_tokens"] = held
spec.loader.exec_module(held)


@pytest.mark.parametrize("name", list(held.TASKS))
def test_no_training_answer_is_a_held_token_and_tests_ask_them(name: str) -> None:
    task = held.make_task(name)
    rng = np.random.default_rng(0)
    hidden = set(task.held)
    heard = set()
    for _ in range(30):
        for m in task.episode(rng, train=True):
            heard |= {int(i) for i in np.argmax(m.x, axis=1) if i < task.vocab}
            if m.label is not None:
                assert not hidden & {int(v) for v in m.label}
    if task.held_mode == "unanswered":
        assert hidden <= heard
    else:
        assert not hidden & heard
    asked = set()
    for _ in range(60):
        for m in task.episode(rng, train=False):
            if m.label is not None:
                asked |= {int(v) for v in m.label}
    assert hidden & asked


def test_the_cut_silences_the_association_onto_the_tokens_alone() -> None:
    task = held.make_task("nback")
    brain = held.make_brain(task, "gain", 0)
    graph = brain.connectome
    pre, post = np.asarray(graph.pre), np.asarray(graph.post)
    motor = np.asarray(graph.populations["motor"])[:task.actions]
    cut = np.isin(pre, np.asarray(graph.populations["association"])) & np.isin(post, motor)
    before = np.array(brain.learner.brain.efficacy)
    held.silence_association_vote(brain, task.actions)
    after = np.asarray(brain.learner.brain.efficacy)
    assert cut.any() and not after[cut].any()
    assert np.array_equal(after[~cut], before[~cut])


def test_a_life_reads_held_and_taught_answers_and_the_vote() -> None:
    row = held.run_life({"task": "nback", "arm": "gain", "founder": 0, "lessons": 5,
                         "tests": 3, "support": ["no_association_vote"]})
    test = row["test"]
    assert {"held", "taught"} <= set(test)
    assert len(row["gains"]) == 2 and test["held"]["best_port"] in ("recall0", "next0")
    missed = test["held"].get("vote_missed")
    if missed is not None:
        assert {"answer_association", "chosen_association", "answer_gain"} <= set(missed)
    assert "held" in row["no_association_vote"]
