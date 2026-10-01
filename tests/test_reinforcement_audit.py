"""Immediate-reward replay must not depend on irrelevant future queries."""

import pytest

from cadence import Brain, Cortex, Reinforcement


def learner(*, vector=False, discount=0.0, horizon=1):
    c = Cortex(seed=7)
    x = c.input("x", shape=1)
    if vector:
        q = c.column("values", patches=2, inputs=x)
        c.output("left", shape=1, reads=q, indices=(0,))
        c.output("right", shape=1, reads=q, indices=(1,))
        options = {"action_input": None, "value_output": ("left", "right")}
    else:
        a = c.input("action", shape=2)
        q = c.column("q", patches=1, inputs=(x, a))
        c.output("value", shape=1, reads=q)
        options = {}
    return Reinforcement(
        c.build(),
        actions=2,
        discount=discount,
        exploration=0,
        credit_horizon=horizon,
        batch_size=1,
        capacity=8,
        **options,
    )


def acknowledge(agent, reward, following, *, terminal=False):
    decision = agent.act({"x": [0.2]}, explore=False)
    assert decision["accepted"]
    receipt = agent.feedback(
        reward,
        following,
        decision_id=decision["decision_id"],
        executed_action=decision["action"],
        terminal=terminal,
        learn=False,
    )
    assert receipt["stored"] and not receipt["accepted"]
    return decision


def test_real_zero_budget_immediate_fit_cannot_be_blocked_by_future_refusal():
    agent = learner()
    decision = agent.act({"x": [0.2]}, explore=False)
    # This actual bounded reward gives a fully qualified current-row fit. A
    # different future input is deliberately not settled at the current state.
    target = decision["settlement"]["predictions"][0]
    reward = target / agent.config["value_scale"]
    agent.feedback(
        reward,
        {"x": [-0.9]},
        decision_id=decision["decision_id"],
        executed_action=decision["action"],
        learn=False,
    )
    snapshot = agent.brain.snapshot()
    reference = Brain.from_snapshot(snapshot)
    action = tuple(float(i == decision["action"]) for i in range(2))
    actual_target = agent.config["value_scale"] * reward
    fit = reference.observe_batch(
        [({"x": [0.2], "action": action}, {"value": (actual_target,)})],
        source="estimate",
        budget=0,
    )
    assert fit["accepted"] and fit["stationarity"] == 0
    result = agent.replay(budget=0)
    assert result["accepted"] and result["qualified"]
    assert result["credit_stops"] == ("zero_discount",)
    assert result["credit_horizons"] == (1,)
    assert result["targets"] == (actual_target,)
    assert result["work"] == fit["work"]
    assert agent.brain.snapshot() == reference.snapshot()


@pytest.mark.parametrize("vector", [False, True])
@pytest.mark.parametrize("horizon", [1, 4])
@pytest.mark.parametrize("terminal", [False, True])
def test_zero_future_weight_never_queries_future_but_whole_fit_qualifies(
    monkeypatch,
    vector,
    horizon,
    terminal,
):
    agent = learner(vector=vector, horizon=horizon)
    acknowledge(agent, 0.25, None if terminal else {"x": [1e308]}, terminal=terminal)
    live = agent.brain.state

    def no_bootstrap(*_args, **_kwargs):
        pytest.fail("zero-discount replay queried a future state")

    monkeypatch.setattr(agent.brain, "settle", no_bootstrap)
    result = agent.replay()
    assert result["accepted"] and result["qualified"]
    assert result["source"] == "estimate"
    assert result["targets"] == (0.225,)
    assert result["credit_horizons"] == (1,)
    assert result["credit_stops"] == ("terminal" if terminal else "zero_discount",)
    assert result["work"]["evaluations"] > 0
    assert agent.brain.state == live


def test_positive_discount_still_requires_qualified_future_and_is_atomic():
    agent = learner(discount=0.5)
    acknowledge(agent, 0.25, {"x": [-0.9]})
    before = agent.snapshot()
    result = agent.replay(budget=0)
    assert not result["accepted"] and result["reason"] == "bootstrap_refused"
    assert result["work"]["evaluations"] > 0
    assert agent.snapshot() == before


def test_zero_discount_fit_refusal_preserves_feedback_for_retry_and_snapshot():
    agent = learner()
    decision = agent.act({"x": [0.2]}, explore=False)
    actual = 1 - decision["action"]
    before_fit = agent.brain.snapshot()
    first = agent.feedback(
        0.8,
        {"x": [1e308]},
        decision_id=decision["decision_id"],
        executed_action=actual,
        budget=0,
    )
    assert first["stored"] and not first["accepted"]
    assert first["reason"] != "bootstrap_refused"
    assert agent.brain.snapshot() == before_fit
    saved = agent.snapshot()
    clone = Reinforcement.from_snapshot(saved)
    assert clone.snapshot() == saved
    retry = clone.feedback(
        0.8,
        {"x": [1e308]},
        decision_id=decision["decision_id"],
        executed_action=actual,
    )
    assert retry["duplicate"] and retry["work"] == {}
    assert clone.snapshot() == saved
    left, right = agent.replay(), clone.replay()
    assert left == right and left["accepted"]
    assert left["targets"] == (0.7200000000000001,)
    assert left["credit_stops"] == ("zero_discount",)
    assert agent.snapshot() == clone.snapshot()
    # The acknowledged alternative owns the actual teaching target.
    record = clone._records[-1]
    assert record[1] == actual
