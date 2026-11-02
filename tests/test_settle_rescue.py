"""An answer whose solve does not qualify within ``free_steps`` continues once with halved
integration steps before the brain refuses it (cadence#239: trained recall brains refused acts
whose settle needed more sweeps at the full step than the budget left them)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import cadence as cd

X = np.random.default_rng(0).random((4, 6))


def brain(budget: int | None = None) -> cd.Brain:
    composed = cd.Brain.compose(6, 4, seed=0)
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
