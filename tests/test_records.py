"""Temporal qualification, causal wake and atomic historical-record custody."""

import math
import sys
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest

from cadence import _repair as R
from cadence._records import RecordSession


def owner(**kwargs):
    # Fast forecast x0 reads live u0 and historical slow x1. Slow x1 reads
    # historical u0, actual ACK u1, and recorded fast error. No slow clamp.
    graph = R.Graph(
        2,
        2,
        (
            ("input", 0, 0),
            ("state", 1, 0),
            ("input", 0, 1),
            ("input", 1, 1),
            ("residual", 0, 1),
        ),
    )
    return RecordSession(
        graph,
        (0,),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.7, 0.2, 0.2, 0.5, 0.3),
        (0.0, 0.0),
        forecast_indices=(0,),
        outcome_inputs=(1,),
        **kwargs,
    )


def check(s):
    active = s._active
    cache = active._cache
    ref = R._evaluate(
        s.graph,
        cache.inputs,
        cache.state,
        s.weights,
        s.biases,
        active.state_prior,
        None,
        None,
        0.1,
        parameter_gradients=False,
    )
    residual = R._stationarity(
        cache.state,
        s.weights,
        s.biases,
        ref,
        {},
        False,
        active.state_bound,
        active.parameter_bound,
    )
    result = s.result()
    assert not active._clamps
    assert result["stationarity"] == residual <= active.tolerance
    assert result["energy"] == ref["energy"]
    assert result["qualified"] and len(result["state"]) == s.graph.n_patches
    return result


def wait_proposal(s):
    s._pending[1].result(timeout=3)
    return s.poll()


def test_changing_familiar_inputs_zero_slow_jobs_full_global_qualification():
    s = owner()
    try:
        answers = []
        for u in (0.2, -0.4, 0.7, -0.1, 0.0):
            output = s.step((u,))
            answers.append(output["forecast"])
            check(s)
            # Declared deterministic fixture, independently computed from input;
            # this is a hand-set software test, not acquired skill evidence.
            row = s.acknowledge(output["ticket"], (math.tanh(0.7 * u) / 1.01,))
            assert not row["wake"]
        result = s.result()
        assert len(set(answers)) == 5
        assert result["counts"]["requests"] == 0
        assert result["jobs"]["foreground"][1] == 0
        assert result["record_age_events"] == 5
        assert result["original_graph_sha256"] != result["lowered_graph_sha256"]
        assert result["jobs"]["foreground"][0] > 0
    finally:
        s.close()


def test_paused_proposal_allows_multiple_fast_ack_cycles_without_starvation(
    monkeypatch,
):
    s = owner()
    entered, release = threading.Event(), threading.Event()
    original = s._propose

    def paused(request):
        entered.set()
        assert release.wait(3)
        return original(request)

    monkeypatch.setattr(s, "_propose", paused)
    try:
        output = s.step((0.1,))
        s.acknowledge(output["ticket"], (0.7,))
        assert entered.wait(1)
        request = s._pending[0]
        for u in (-0.3, 0.4, -0.2):
            output = s.step((u,))
            s.acknowledge(output["ticket"], output["forecast"])
            result = check(s)
            assert result["pending"] and result["record_version"] == 0
            assert [r["event"] for r in result["demand"]] == [1]
        assert request.record.inputs == (0.1, 0.7)
        release.set()
        assert wait_proposal(s)["status"] == "committed"
        after = check(s)
        assert after["record_version"] == 1 and after["record_event"] == 1
        assert after["record_age_events"] == 3
        assert [r["event"] for r in after["demand"]] == [
            1
        ]  # commit alone does not clear demand
        assert s._inputs[0] == -0.2  # latest fast input, not worker's old input
        output = s.step((0.1,))
        s.acknowledge(output["ticket"], output["forecast"])
        assert s.result()["demand"] == ()
    finally:
        release.set()
        s.close()


def test_accurate_forecast_terminal_unmet_goal_wakes_without_surprise():
    s = owner(value_index=0)
    try:
        issued = s.step((0.2,), required_value=0.8)
        row = s.acknowledge(
            issued["ticket"],
            issued["forecast"],
            task_value=issued["ticket"].predicted_value,
            terminal=True,
        )
        assert row["surprise"] == 0.0 and row["goal_deficit"]
        assert row["predicted_value"] == issued["forecast"][0]
        assert s.result()["counts"]["requests"] == 1
        assert wait_proposal(s)["status"] == "committed"
        assert s.result()["demand"]
    finally:
        s.close()


def test_duplicate_foreign_and_invalid_ack_do_not_replace_issued_forecast():
    s = owner()
    try:
        issued = s.step((0.3,))
        ticket = issued["ticket"]
        with pytest.raises(ValueError, match="foreign"):
            s.acknowledge(replace(ticket, forecast=(0.8,)), (0.8,))
        with pytest.raises(ValueError):
            s.acknowledge(ticket, (float("nan"),))
        with pytest.raises(ValueError, match="outstanding"):
            s.step((0.2,))
        row = s.acknowledge(ticket, ticket.forecast)
        assert row["forecast"] == ticket.forecast
        with pytest.raises(ValueError, match="duplicate"):
            s.acknowledge(ticket, ticket.forecast)
        assert len(s.ledger()) == 1
    finally:
        s.close()


def test_newer_mismatch_demand_survives_older_valid_record_commit(monkeypatch):
    s = owner()
    entered, release = threading.Event(), threading.Event()
    original = s._propose

    def paused(request):
        entered.set()
        assert release.wait(3)
        return original(request)

    monkeypatch.setattr(s, "_propose", paused)
    try:
        output = s.step((0.1,))
        s.acknowledge(output["ticket"], (0.7,))
        assert entered.wait(1)
        output = s.step((-0.2,))
        s.acknowledge(output["ticket"], (-0.7,))
        release.set()
        assert wait_proposal(s)["status"] == "committed"
        assert s.result()["record_event"] == 1
        assert [r["event"] for r in s.result()["demand"]] == [1, 2]
        assert s._pending[0].event == 2
        assert wait_proposal(s)["status"] == "committed"
        assert s.result()["record_event"] == 2
        assert [r["event"] for r in s.result()["demand"]] == [1, 2]
        check(s)
    finally:
        release.set()
        s.close()


def test_goal_aba_stales_proposal_without_changing_committed_record(monkeypatch):
    s = owner()
    entered, release = threading.Event(), threading.Event()
    original = s._propose

    def paused(request):
        entered.set()
        assert release.wait(3)
        return original(request)

    monkeypatch.setattr(s, "_propose", paused)
    try:
        output = s.step((0.1,))
        s.acknowledge(output["ticket"], (0.7,))
        assert entered.wait(1)
        for goal in (1, 0):
            output = s.step((0.2,), goal_version=goal)
            s.acknowledge(output["ticket"], output["forecast"])
        release.set()
        assert wait_proposal(s)["status"] == "stale"
        assert s.result()["record_version"] == 0
        assert [r["event"] for r in s.result()["demand"]] == [1]
        check(s)
    finally:
        release.set()
        s.close()


def test_refused_slow_proposal_preserves_demand_and_active_state(monkeypatch):
    s = owner()
    monkeypatch.setattr(s, "_propose", lambda _request: {"qualified": False})
    try:
        output = s.step((0.3,))
        before = check(s)
        s.acknowledge(output["ticket"], (0.9,))
        assert wait_proposal(s)["status"] == "refused"
        after = check(s)
        assert after["state"] == before["state"]
        assert after["record_version"] == 0 and [
            r["event"] for r in after["demand"]
        ] == [1]
        assert s.result()["counts"]["requests"] == 1
        s.poll()
        assert s.result()["counts"]["requests"] == 1
    finally:
        s.close()


def test_ledger_bound_refuses_next_issue_before_losing_outcome_and_close():
    s = owner(capacity=1)
    output = s.step((0.2,))
    s.acknowledge(output["ticket"], output["forecast"])
    with pytest.raises(ValueError, match="full"):
        s.step((0.3,))
    assert len(s.ledger()) == 1
    result = s.close()
    assert not result["qualified"] and result["state"] is None
    with pytest.raises(ValueError, match="closed"):
        s.step((0.2,))


def test_live_coupling_within_each_block_is_not_lowered():
    graph = R.Graph(
        2,
        4,
        (
            ("input", 0, 0),
            ("state", 0, 1),
            ("input", 1, 2),
            ("residual", 2, 3),
            ("residual", 1, 2),
            ("state", 3, 0),
        ),
    )
    s = RecordSession(
        graph,
        (0, 1),
        (0.0, 0.0),
        (0.0,) * 4,
        (0.2,) * 6,
        (0.0,) * 4,
        forecast_indices=(1,),
        outcome_inputs=(1,),
    )
    try:
        assert s.graph.edges[1] == ("state", 0, 1)
        assert s.graph.edges[3] == ("residual", 2, 3)
        assert s.graph.edges[4][0] == s.graph.edges[5][0] == "input"
        check(s)
    finally:
        s.close()


def test_fast_change_reuses_exact_cache_without_any_slow_prediction(monkeypatch):
    import cadence._incremental as module

    s = owner()
    visited = []

    def tanh(value):
        visited.append(sys._getframe(1).f_locals["target"])
        return math.tanh(value)

    def forbidden(*_args, **_kwargs):
        pytest.fail("foreground reconstructed a full cache")

    monkeypatch.setattr(module.ActivityCache, "__init__", forbidden)
    monkeypatch.setattr(
        module,
        "math",
        SimpleNamespace(tanh=tanh, fsum=math.fsum, isfinite=math.isfinite),
    )
    try:
        setups = s.result()["work"]["setups"]
        for value in (0.15, -0.3, 0.5):
            output = s.step((value,))
            s.acknowledge(output["ticket"], (math.tanh(0.7 * value) / 1.01,))
            check(s)
        result = s.result()
        assert result["work"]["setups"] == setups
        assert visited and set(visited) == {0}
        assert result["jobs"]["foreground"][1] == 0
        assert result["work"]["certificate_checks"] > 0
        assert result["work"]["forks"] > 0
    finally:
        s.close()


def test_records_config_is_fixed_and_receipts_reject_mutation_and_commit():
    s = owner()
    try:
        for name in ("weights", "biases", "ports", "graph", "threshold"):
            with pytest.raises(AttributeError):
                setattr(s, name, getattr(s, name))
        with pytest.raises(TypeError):
            s.config["tolerance"] = 1.0
        output = s.step((0.2,))
        receipt = output["certificate"]
        assert s.is_current(receipt)
        assert not s.is_current({**receipt, "state": (0.8, 0.8)})
        s.acknowledge(output["ticket"], (0.7,))
        assert not s.is_current(receipt)
        assert wait_proposal(s)["status"] == "committed"
        assert not s.is_current(receipt)
        assert s.is_current(s.result())
    finally:
        s.close()


def test_terminal_deficit_requires_matching_terminal_recovery_not_quiet_sample():
    s = owner()
    try:
        output = s.step((0.2,), required_value=0.6)
        s.acknowledge(
            output["ticket"], output["forecast"], task_value=0.0, terminal=True
        )
        wait_proposal(s)
        for value, terminal, reward in ((0.4, True, 0.9), (0.2, False, None)):
            output = s.step((value,), required_value=0.6)
            s.acknowledge(
                output["ticket"],
                output["forecast"],
                task_value=reward,
                terminal=terminal,
            )
            assert len(s.result()["demand"]) == 1
        output = s.step((0.2,), required_value=0.6)
        s.acknowledge(
            output["ticket"], output["forecast"], task_value=0.9, terminal=True
        )
        assert not s.result()["demand"]
    finally:
        s.close()


def test_requirement_change_same_goal_version_stales_pending_proposal(monkeypatch):
    s = owner()
    entered, release = threading.Event(), threading.Event()
    original = s._propose

    def paused(request):
        entered.set()
        assert release.wait(3)
        return original(request)

    monkeypatch.setattr(s, "_propose", paused)
    try:
        output = s.step((0.2,), required_value=0.6)
        s.acknowledge(output["ticket"], (0.7,))
        assert entered.wait(1)
        output = s.step((0.2,), required_value=0.9)
        s.acknowledge(output["ticket"], output["forecast"])
        release.set()
        assert wait_proposal(s)["status"] == "stale"
        assert s.result()["record_version"] == 0
        assert s.result()["demand"][0]["required"] == 0.6
    finally:
        release.set()
        s.close()


@pytest.mark.parametrize("failure", ["refused", "error"])
def test_newer_terminal_evidence_starts_after_older_failure_without_another_ack(
    monkeypatch, failure
):
    s = owner()
    entered, release = threading.Event(), threading.Event()
    original = s._propose

    def first_fails(request):
        if request.event == 1:
            entered.set()
            assert release.wait(3)
            if failure == "error":
                raise RuntimeError("retained worker failure")
            return {"qualified": False}
        return original(request)

    monkeypatch.setattr(s, "_propose", first_fails)
    try:
        output = s.step((0.1,))
        s.acknowledge(output["ticket"], (0.7,))
        assert entered.wait(1)
        output = s.step((-0.2,))
        s.acknowledge(output["ticket"], (-0.7,))
        release.set()
        if failure == "error":
            with pytest.raises(RuntimeError):
                s._pending[1].result(timeout=3)
            assert s.poll()["status"] == "error"
        else:
            assert wait_proposal(s)["status"] == "refused"
        assert s._pending[0].event == 2
        assert wait_proposal(s)["status"] == "committed"
        assert {r["event"] for r in s.result()["demand"]} == {1, 2}
        output = s.step((-0.2,))
        s.acknowledge(output["ticket"], output["forecast"])
        assert {r["event"] for r in s.result()["demand"]} == {1}
    finally:
        release.set()
        s.close()


def test_close_observes_worker_failure_once_and_ledger_preserves_issued_origin(
    monkeypatch,
):
    s = owner()

    def failed(_request):
        raise RuntimeError("failed private solve")

    monkeypatch.setattr(s, "_propose", failed)
    output = s.step((0.2,))
    ticket = output["ticket"]
    s.acknowledge(ticket, (0.7,))
    result = s.close()
    assert result["unknown_work"] and result["counts"]["worker_errors"] == 1
    assert result["counts"]["proposal_returns"] == 1
    assert s.close()["counts"]["worker_errors"] == 1
    row = s.ledger()[0]
    assert row["issued"]["inputs"] == ticket.inputs
    assert row["issued"]["errors"] == ticket.errors
    row["issued"]["inputs"] = (99.0, 99.0)
    assert s.ledger()[0]["issued"]["inputs"] == ticket.inputs


def test_commit_refusal_preserves_whole_prior_record_state_and_demand(monkeypatch):
    s = owner()
    solve = s._solve

    def refuse_commit(record, inputs, state, role, **kwargs):
        session, result = solve(record, inputs, state, role, **kwargs)
        return session, {**result, "qualified": False} if role == "commit" else result

    monkeypatch.setattr(s, "_solve", refuse_commit)
    try:
        output = s.step((0.2,))
        before = s.result()
        record = s._record
        s.acknowledge(output["ticket"], (0.7,))
        assert wait_proposal(s)["status"] == "refused_commit"
        after = check(s)
        assert s._record is record and after["state"] == before["state"]
        assert after["demand"] and after["counts"]["commits"] == 0
        assert after["work"]["edge_visits"] > before["work"]["edge_visits"]
    finally:
        s.close()


def test_failed_background_start_runs_no_unowned_proposal(monkeypatch):
    s = owner()
    start = threading.Thread.start
    called = []
    monkeypatch.setattr(s, "_propose", lambda request: called.append(request))

    def started_then_failed(thread):
        start(thread)
        raise RuntimeError("late thread startup failure")

    try:
        output = s.step((0.2,))
        with monkeypatch.context() as patch:
            patch.setattr(threading.Thread, "start", started_then_failed)
            s.acknowledge(output["ticket"], (0.7,))
        result = s.close()
        assert result["unknown_work"] and result["counts"]["submit_failures"] == 1
        assert result["counts"]["requests"] == 1
        assert result["counts"]["proposal_returns"] == 0
        assert result["demand"] and called == []
    finally:
        s.close()


def test_setup_error_marks_unknown_and_preserves_active_record(monkeypatch):
    import cadence._records as module

    s = owner()
    active = s._active
    construct = module._Tracked

    def failed(*args, **kwargs):
        created = construct(*args, **kwargs)
        created.close()
        raise RuntimeError("setup work before absent return")

    monkeypatch.setattr(module, "_Tracked", failed)
    try:
        with pytest.raises(RuntimeError, match="setup work"):
            s._solve(s._record, s._inputs, (0.0, 0.0), "proposal")
        assert s._active is active and s.result()["unknown_work"]
        check(s)
    finally:
        s.close()


def test_commit_exception_preserves_state_and_allows_newer_queued_evidence(monkeypatch):
    s = owner()
    propose, solve = s._propose, s._solve
    entered, release = threading.Event(), threading.Event()

    def paused(request):
        if request.event == 1:
            entered.set()
            assert release.wait(3)
        return propose(request)

    def failed_commit(record, inputs, state, role, **kwargs):
        if role == "commit" and record.event == 1:
            raise ValueError("combined finite contexts overflow")
        return solve(record, inputs, state, role, **kwargs)

    monkeypatch.setattr(s, "_propose", paused)
    monkeypatch.setattr(s, "_solve", failed_commit)
    try:
        output = s.step((0.1,))
        s.acknowledge(output["ticket"], (0.7,))
        assert entered.wait(1)
        output = s.step((-0.2,))
        s.acknowledge(output["ticket"], (-0.7,))
        before = s.result()["state"]
        release.set()
        assert wait_proposal(s)["status"] == "commit_error"
        assert s.result()["record_version"] == 0
        assert s.result()["state"] == before
        assert s._pending[0].event == 2
        assert wait_proposal(s)["status"] == "committed"
        assert {r["event"] for r in s.result()["demand"]} == {1, 2}
    finally:
        release.set()
        s.close()


def test_retained_issued_errors_reconstruct_from_original_lowered_context():
    s = owner()
    try:
        output = s.step((0.3,))
        s.acknowledge(output["ticket"], (0.7,))
        row = s.ledger()[0]
        issued, record = row["issued"], row["record_at_issue_details"]
        extra = tuple(
            record["inputs"][i]
            if kind == "input"
            else record["state"][i]
            if kind == "state"
            else record["errors"][i]
            for kind, i in s.ports
        )
        evaluated = R._evaluate(
            s.graph,
            issued["inputs"] + extra,
            issued["state"],
            s.weights,
            s.biases,
            s._active.state_prior,
            None,
            None,
            0.1,
            parameter_gradients=False,
        )
        assert evaluated["errors"] == issued["errors"]
        assert row["actual"] != issued["forecast"]
        assert issued["errors"] != tuple(
            a - b for a, b in zip(row["actual"], issued["forecast"], strict=True)
        )
    finally:
        s.close()
