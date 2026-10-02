"""Factual outcome custody and retention across private asynchronous repair."""

import threading
import time

import pytest

from cadence import Cortex
from cadence._experience import ForecastSession


def brain():
    c = Cortex(seed=11, tolerance=1e-5)
    u = c.input("sense", shape=1)
    p = c.column("routine", patches=1, inputs=u)
    c.output("next", shape=1, reads=p)
    return c.build()


def drain(owner):
    end = time.monotonic() + 5
    while owner.inspect()["active"]:
        owner.poll()
        if time.monotonic() > end:
            pytest.fail("factual learner failed to complete in test bound")
        time.sleep(0.001)


def issue(owner, value=0.0):
    result = owner.step({"sense": [value]}, seconds=2)
    assert result["accepted"], result
    return result


def test_real_disturbance_is_learned_retained_and_then_costs_no_replay():
    original = brain()
    # Routine competence must be acquired, not the accidental zero prediction
    # of a newborn queried with zero input and zero bias.
    for _ in range(4):
        assert original.observe({"sense": [0.0]}, {"next": [0.2]})["accepted"]
    checkpoint = original.snapshot()
    owner = ForecastSession(original, forecasts=("next",), threshold=0.03, batch_size=1)
    try:
        for _ in range(3):
            result = issue(owner)
            assert abs(result["outputs"]["next"][0] - 0.2) < 0.03
            owner.feedback(result["decision_id"], {"next": [0.2]})
        assert owner.inspect()["counts"]["scheduled"] == 0
        first = issue(owner)
        error_before = abs(first["outputs"]["next"][0] - 0.4)
        ack = owner.feedback(first["decision_id"], {"next": [0.4]})
        assert ack["demand"] == ["surprise"]
        drain(owner)
        # An accepted fit is not an actual observation confirming recovery.
        assert owner.inspect()["demand"] == ["surprise"]
        for _ in range(8):
            result = issue(owner)
            owner.feedback(result["decision_id"], {"next": [0.4]})
            drain(owner)
            if not owner.inspect()["demand"]:
                break
        assert abs(result["outputs"]["next"][0] - 0.4) < 0.03 < error_before
        assert owner.inspect()["demand"] == []
        scheduled = owner.inspect()["counts"]["scheduled"]
        submissions = result["activity"]["counts"]["submissions"]
        for _ in range(4):
            result = issue(owner)
            assert abs(result["outputs"]["next"][0] - 0.4) < 0.03
            owner.feedback(result["decision_id"], {"next": [0.4]})
        assert owner.inspect()["counts"]["scheduled"] == scheduled
        assert result["activity"]["counts"]["submissions"] == submissions
        assert owner.inspect()["learning_work"]["edge_visits"] > 0
        assert original.snapshot() == checkpoint
    finally:
        assert owner.close()["work_complete"]


def test_background_fit_does_not_block_foreground_or_overwrite_new_activity(
    monkeypatch,
):
    owner = ForecastSession(brain(), forecasts=("next",), batch_size=1)
    entered, release = threading.Event(), threading.Event()
    fit = owner._fit

    def delayed(*args):
        entered.set()
        assert release.wait(3)
        return fit(*args)

    monkeypatch.setattr(owner, "_fit", delayed)
    try:
        first = issue(owner)
        owner.feedback(first["decision_id"], {"next": [0.5]})
        assert entered.wait(1)
        second = issue(owner, 0.6)
        live = owner._brain.state
        assert owner.inspect()["active"]
        release.set()
        drain(owner)
        assert owner._brain.state == live
        assert owner.inspect()["generation"] == 1
        # A prior-generation forecast matching by chance cannot clear current
        # demand after a different parameter generation has been installed.
        owner.feedback(second["decision_id"], second["outputs"])
        assert owner.inspect()["demand"] == ["surprise"]
    finally:
        release.set()
        owner.close()


def test_outcomes_are_atomic_and_latest_retry_is_idempotent():
    owner = ForecastSession(brain(), forecasts=("next",), learning_budget=0)
    try:
        result = issue(owner)
        baseline = owner.inspect()
        for values in (
            {"next": [float("nan")]},
            {"next": [2.0]},
            {},
            {"next": [0.2, 0.3]},
        ):
            with pytest.raises(ValueError):
                owner.feedback(result["decision_id"], values)
            assert owner.inspect() == baseline
        with pytest.raises(ValueError):
            owner.feedback(result["decision_id"] + 1, {"next": [0.4]})
        ack = owner.feedback(result["decision_id"], {"next": [0.4]})
        assert not ack["duplicate"]
        duplicate = owner.feedback(result["decision_id"], {"next": [0.4]})
        assert duplicate["duplicate"] and owner.inspect()["evidence"] == 1
        with pytest.raises(ValueError):
            owner.feedback(result["decision_id"], {"next": [0.3]})
        drain(owner)
        assert owner.inspect()["counts"]["fit_refused"] == 1
        assert owner.inspect()["evidence"] == 1
        assert owner.inspect()["demand"] == ["surprise"]
        for _ in range(3):
            owner.poll()
        assert owner.inspect()["counts"]["scheduled"] == 1
    finally:
        owner.close()


def test_refused_activity_preserves_accepted_brain_and_does_not_issue():
    owner = ForecastSession(brain(), forecasts=("next",))
    try:
        before = owner._brain.snapshot()
        result = owner.step({"sense": [0.7]}, budget=0)
        assert not result["accepted"] and result["decision_id"] is None
        assert owner._brain.snapshot() == before
        assert owner.inspect()["pending"] is None
        assert issue(owner)["decision_id"] == 1
    finally:
        assert owner.close()["work_complete"]


def test_no_future_clamp_even_through_output_alias():
    c = Cortex()
    u = c.input("sense", shape=1)
    p = c.column("routine", patches=1, inputs=u)
    c.output("next", shape=1, reads=p)
    c.output("alias", shape=1, reads=p)
    owner = ForecastSession(c.build(), forecasts=("next",))
    try:
        with pytest.raises(ValueError, match="unclamped"):
            owner.step({"sense": [0.0]}, targets={"alias": [0.4]})
        assert owner.inspect()["counts"]["issued"] == 0
    finally:
        owner.close()


def test_missed_terminal_goal_recruits_even_with_perfect_predictions():
    owner = ForecastSession(brain(), forecasts=("next",), goal=1.0, batch_size=1)
    try:
        result = issue(owner)
        owner.feedback(
            result["decision_id"], result["outputs"], reward=-0.2, terminal=True
        )
        assert owner.inspect()["demand"] == ["terminal_failure"]
        drain(owner)
        assert owner.inspect()["demand"] == ["terminal_failure"]
        with pytest.raises(ValueError, match="start_episode"):
            issue(owner)
        owner.start_episode()
        assert owner.inspect()["demand"] == ["terminal_failure"]
        result = issue(owner)
        owner.feedback(
            result["decision_id"], result["outputs"], reward=1.0, terminal=True
        )
        assert owner.inspect()["demand"] == []
        assert owner.inspect()["counts"]["scheduled"] == 1
    finally:
        owner.close()


def test_body_mutation_cannot_rewrite_forecast_or_witness(monkeypatch):
    owner = ForecastSession(brain(), forecasts=("next",), learning_budget=0)
    try:
        inputs = {"sense": [0.0]}
        result = owner.step(inputs)
        inputs["sense"][0] = 0.9
        result["outputs"]["next"] = (0.4,)
        actual = {"next": [0.4]}
        feedback = owner.feedback(result["decision_id"], actual)
        actual["next"][0] = -0.8
        assert feedback["surprise"] == pytest.approx(0.4)
        assert owner._rows[0][1] == {"sense": (0.0,)}
        assert owner._rows[0][2] == {"next": (0.4,)}
    finally:
        owner.close()


def test_unrelated_quiet_context_cannot_erase_an_unrepaired_surprise():
    owner = ForecastSession(brain(), forecasts=("next",), learning_budget=0)
    try:
        first = issue(owner)
        owner.feedback(first["decision_id"], {"next": [0.5]})
        drain(owner)
        other = issue(owner, 0.7)
        owner.feedback(other["decision_id"], other["outputs"])
        assert owner.inspect()["demand"] == ["surprise"]
        assert owner.inspect()["unresolved_contexts"] == 1
    finally:
        owner.close()


def test_unresolved_overflow_never_reports_recovery():
    owner = ForecastSession(
        brain(), forecasts=("next",), learning_budget=0, capacity=1, batch_size=1
    )
    try:
        for value in (0.0, 0.8):
            result = issue(owner, value)
            owner.feedback(result["decision_id"], {"next": [0.9]})
            drain(owner)
        result = issue(owner, 0.8)
        owner.feedback(result["decision_id"], result["outputs"])
        assert owner.inspect()["deficit_overflow"]
        assert owner.inspect()["demand"] == ["surprise"]
        assert len(owner._rows) == 1
    finally:
        owner.close()


def test_failed_submission_keeps_fact_and_is_not_retried_without_new_evidence(
    monkeypatch,
):
    owner = ForecastSession(brain(), forecasts=("next",))

    def rejected(*args):
        raise RuntimeError("test worker unavailable")

    monkeypatch.setattr(owner._pool, "submit", rejected)
    try:
        result = issue(owner)
        ack = owner.feedback(result["decision_id"], {"next": [0.4]})
        for _ in range(4):
            owner.poll()
        status = owner.inspect()
        assert ack["evidence_id"] == status["evidence"] == 1
        assert status["counts"]["scheduled"] == status["counts"]["submit_errors"] == 1
        assert status["receipts"][0]["job"]["witnesses"][0][3]["forecast"] == {
            "next": (0.0,)
        }
        assert status["demand"] == ["surprise"]
        assert status["unknown_work"] and status["learning_disabled"]
    finally:
        owner.close()


def test_repeated_refusals_reap_owned_caches_and_retain_aggregate_cost():
    owner = ForecastSession(brain(), forecasts=("next",))
    try:
        for _ in range(10):
            assert not owner.step({"sense": [0.6]}, budget=0)["accepted"]
            assert len(owner._retired) <= 1
        assert owner.inspect()["retired_activity_work"]["setups"] == 9
    finally:
        result = owner.close()
        assert result["work_complete"]
        assert result["retired_activity_work"]["setups"] == 10
        assert result["retired_pending"] == 0


def test_configuration_and_invalid_empty_values_do_not_mutate_owner():
    owner = ForecastSession(brain(), forecasts=("next",))
    try:
        for name in owner._FROZEN:
            with pytest.raises(AttributeError):
                setattr(owner, name, None)
            with pytest.raises(AttributeError):
                delattr(owner, name)
        for targets in (False, [], "", 0):
            with pytest.raises(ValueError):
                owner.step({"sense": [0.0]}, targets=targets)
        assert owner.inspect()["counts"]["issued"] == 0
    finally:
        owner.close()


@pytest.mark.parametrize("fails", (False, True))
def test_close_counts_drained_learning_without_admitting_it(monkeypatch, fails):
    owner = ForecastSession(brain(), forecasts=("next",), batch_size=1)
    if fails:

        def failed(*args):
            raise RuntimeError("intentional worker failure")

        monkeypatch.setattr(owner, "_fit", failed)
    result = issue(owner)
    owner.feedback(result["decision_id"], {"next": [0.4]})
    closed = owner.close()
    assert closed["work_complete"]
    assert closed["generation"] == 0
    assert closed["counts"]["fit_errors" if fails else "discarded_on_close"] == 1
    assert closed["unknown_work"] is fails
    assert closed["evidence"] == 1
