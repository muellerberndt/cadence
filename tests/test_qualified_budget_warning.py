"""A qualified nudged budget below the free budget warns and names itself on refusal.

Issue 123: `Brain.compose` keeps `nudged_steps=12` as the finite teaching law,
but under `qualified=True` that number becomes a settle budget and every
realistic lesson is refused with no hint that the budget, not the data, is the
cause. The configuration now warns at construction and the refusal names the
mismatch. Finite teaching and deliberately small qualified budgets remain.
"""

import inspect
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import cadence as cd


def motor_graph():
    pre, post = np.where(~np.eye(36, dtype=bool))
    connectome = cd.Connectome.from_synapses(36, pre=pre, post=post, sign=np.full(len(pre), -0.5))
    return cd.NeuralGraph(connectome, cd.learning_neuron_model(dt=1))


def test_qualified_config_with_starved_nudged_budget_warns_at_the_call_site() -> None:
    with pytest.warns(RuntimeWarning, match=r"nudged_steps \(12\).*free_steps \(1024\)") as caught:
        line = inspect.currentframe().f_lineno + 1
        config = cd.LearnerConfig(qualified=True, free_steps=1024, nudged_steps=12, tolerance=3e-3)
    assert len(caught) == 1
    assert Path(caught[0].filename).resolve() == Path(__file__).resolve()
    assert caught[0].lineno == line
    assert config.nudged_steps == 12 and config.free_steps == 1024


def test_replacing_a_composed_config_to_qualified_warns_without_changing_budgets() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        brain = cd.Brain.compose(inputs=4, actions=2, modules=(4,), seed=0, recall=False)
    composed = brain.learner.config
    assert composed.free_steps == 1024 and composed.nudged_steps == 12
    with pytest.warns(RuntimeWarning, match="qualified=True"):
        qualified = replace(composed, qualified=True)
    assert qualified.nudged_steps == 12 and qualified.free_steps == 1024
    with pytest.warns(RuntimeWarning, match="qualified=True"):
        restored = cd.LearnerConfig(**qualified.to_dict())
    assert restored == qualified


@pytest.mark.parametrize("free_steps,nudged_steps", [(512, 512), (128, 512), (0, 0), (1, 1)])
def test_qualified_with_comparable_or_equal_budgets_is_silent(free_steps, nudged_steps) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cd.LearnerConfig(
            qualified=True, free_steps=free_steps, nudged_steps=nudged_steps, tolerance=3e-3
        )


def test_finite_teaching_with_a_small_nudged_budget_stays_silent() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cd.LearnerConfig(free_steps=1024, nudged_steps=12, tolerance=3e-3)


def test_refused_nudged_phase_names_the_budget_mismatch() -> None:
    graph = motor_graph()
    with pytest.warns(RuntimeWarning, match="qualified=True"):
        config = cd.LearnerConfig(qualified=True, free_steps=512, nudged_steps=0, tolerance=3e-3)
    learner = cd.Learner(graph, np.arange(36), config)
    with pytest.raises(cd.LearningPhaseError) as caught:
        learner.step(np.full((1, 36), 0.2), np.array([0]))
    failure = caught.value
    assert failure.phase == "nudged"
    assert failure.hint is not None
    assert "nudged budget (0) is below the free budget (512)" in str(failure)
    assert "no learning update applied" in str(failure)


def test_refused_free_phase_and_balanced_budgets_carry_no_hint(monkeypatch) -> None:
    graph = motor_graph()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        config = cd.LearnerConfig(qualified=True, free_steps=0, nudged_steps=0, tolerance=3e-3)
    learner = cd.Learner(graph, np.arange(36), config)
    with pytest.raises(cd.LearningPhaseError) as caught:
        learner.step(np.full((1, 36), 0.2), np.array([0]))
    assert caught.value.phase == "free"
    assert caught.value.hint is None
    assert "below the free budget" not in str(caught.value)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        balanced = cd.LearnerConfig(
            qualified=True, free_steps=128, nudged_steps=128, tolerance=3e-3
        )
    starved = cd.Learner(graph, np.arange(36), balanced)
    solve = starved._qualified_phase

    def exhaust_nudged(drive, state, budget, nudge=None):
        return solve(drive, state, 0 if nudge is not None else budget, nudge)

    monkeypatch.setattr(starved, "_qualified_phase", exhaust_nudged)
    with pytest.raises(cd.LearningPhaseError) as refused:
        starved.step(np.full((1, 36), 0.2), np.array([0]))
    assert refused.value.phase == "nudged"
    assert refused.value.hint is None
    assert "below the free budget" not in str(refused.value)
