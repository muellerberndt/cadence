"""An answer whose solve does not qualify within ``free_steps`` continues once on the damping
ladder before the brain refuses it (cadence#239: trained recall brains refused acts whose
settle needed more sweeps at the full step than the budget left them)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import cadence as cd

X = np.random.default_rng(0).random((4, 6))


def brain(budget: int | None = None) -> cd.Brain:
    composed = cd.Brain.compose(6, 4, seed=0, recall=False)
    if budget is not None:
        composed.learner.config = replace(composed.learner.config, free_steps=budget)
    return composed


def test_a_solve_that_qualifies_takes_no_rescue():
    roomy = brain()
    roomy.act(X, greedy=True)
    assert roomy.last_settlement["qualified"] and "rescue_steps" not in roomy.last_settlement


def test_an_answer_that_does_not_settle_in_its_budget_is_rescued_before_it_is_refused():
    tight = brain(8)
    action = tight.act(X, greedy=True)
    report = tight.last_settlement
    assert report["qualified"] and report["rescue_steps"] > 0
    assert report["steps"] == 8 + report["rescue_steps"]
    roomy = brain()
    np.testing.assert_array_equal(action, roomy.act(X, greedy=True))


def test_an_answer_whose_rescue_fails_too_is_refused():
    hopeless = brain(1)
    with pytest.raises(RuntimeError, match="did not settle within 1 steps and a rescue of as many"):
        hopeless.act(X, greedy=True)
    report = hopeless.last_settlement
    assert not report["qualified"] and report["rescue_steps"] >= 1


def test_a_refused_rescue_reports_the_work_of_both_solves():
    brain = cd.Brain.compose(6, 4, seed=0, recall=False)
    brain.learner.config = replace(brain.learner.config, free_steps=8, tolerance=1e-15)
    with pytest.raises(RuntimeError, match="and a rescue of as many"):
        brain.act(np.random.default_rng(0).random((1, 6)), greedy=True)
    report = brain.last_settlement
    assert report["rescue_steps"] == 8 and report["steps"] == 16
    assert report["residual_checks"] >= 2  # at least one check of each solve


def test_a_state_that_is_not_finite_is_refused_without_a_rescue(monkeypatch):
    from cadence.generic import Brain

    original = Brain._equilibrate

    def blown(self, drive, state, *, budget, tolerance):
        phase = original(self, drive, state, budget=budget, tolerance=tolerance)
        bad = replace(phase.state, v=np.full_like(np.asarray(phase.state.v), np.nan))
        return replace(phase, state=bad, residual=np.full_like(np.asarray(phase.residual), np.inf))

    monkeypatch.setattr(Brain, "_equilibrate", blown)
    brain = cd.Brain.compose(6, 4, seed=0, recall=False)
    with pytest.raises(RuntimeError, match="did not settle") as refused:
        brain.act(np.random.default_rng(0).random((1, 6)), greedy=True)
    assert "rescue" not in str(refused.value)
    assert "rescue_steps" not in brain.last_settlement
