"""Automatic block work, true asynchronous barriers, and whole-state custody."""

import math
import threading
from dataclasses import replace

import pytest

from cadence import Cortex
from cadence import _repair as R
from cadence._attention import ActivitySession


def owner(graph=None, **kwargs):
    graph = graph or R.Graph(1, 1, (("input", 0, 0),))
    return ActivitySession(
        graph,
        tuple((i,) for i in range(graph.n_patches)),
        (0.0,) * graph.n_inputs,
        (0.0,) * graph.n_patches,
        (1.0,) * len(graph.edges),
        (0.0,) * graph.n_patches,
        **kwargs,
    )


def reference(session, result):
    cache = session._cache
    evaluated = R._evaluate(
        session.graph,
        cache.inputs,
        cache.state,
        cache.weights,
        cache.biases,
        session.state_prior,
        None,
        None,
        0.1,
        parameter_gradients=False,
    )
    residual = R._stationarity(
        cache.state,
        cache.weights,
        cache.biases,
        evaluated,
        session._clamps,
        False,
        session.state_bound,
        session.parameter_bound,
    )
    assert result["stationarity"] == residual
    assert result["energy"] == evaluated["energy"]
    if result["qualified"]:
        assert residual <= session.tolerance
        assert result["state"] == cache.state
        assert result["predictions"] == evaluated["predictions"]
        assert result["errors"] == evaluated["errors"]


def test_quiet_has_no_worker_and_changed_input_recruits_without_flags():
    s = owner()
    try:
        started = s.begin((0.0,))
        assert started["started"] == []
        first = s.result()
        assert first["qualified"] and first["counts"]["submissions"] == 0
        s.begin((0.5,))
        assert not s.is_current(first)
        result = s.run_until_complete(seconds=2)
        assert result["qualified"]
        assert result["state"][0] == pytest.approx(math.tanh(0.5) / 1.01, abs=1e-6)
        assert result["counts"]["submissions"] > 0
        assert result["work"]["certificate_checks"] > 0
        reference(s, result)
    finally:
        drained = s.close()
    assert drained["work_complete"]
    assert drained["counts"]["returns"] == drained["counts"]["submissions"]


def test_public_graph_with_observer_all_states_and_original_brain_unchanged():
    c = Cortex(seed=17)
    u = c.input("sense", shape=2)
    p = c.column("prediction", patches=2, inputs=u)
    h = c.observer("evidence", patches=2, observes=p)
    c.output("seen", shape=1, reads=p)
    c.output("answer", shape=2, reads=h)
    brain = c.build()
    before = brain.snapshot()
    s = ActivitySession.from_brain(brain, {"sense": (0.2, -0.3)})
    try:
        flat, fixed = brain._arguments({"sense": (0.2, -0.3)}, {"seen": (0.4,)}, None)
        s.begin(flat, clamps=fixed, budget=512)
        result = s.run_until_complete(seconds=3)
        assert result["qualified"]
        assert len(result["state"]) == 4
        assert result["state"][0] == 0.4
        reference(s, result)
        assert brain.snapshot() == before
    finally:
        s.close()


def test_relevant_slow_worker_blocks_global_publication(monkeypatch):
    s = owner(R.Graph(1, 2, (("input", 0, 0), ("state", 0, 1))))
    entered, release = threading.Event(), threading.Event()
    compute = s._compute

    def delayed(context):
        if context.block == 1:
            entered.set()
            assert release.wait(3)
        return compute(context)

    monkeypatch.setattr(s, "_compute", delayed)
    try:
        s.begin((0.5,), budget=256)
        result = s.run_until_complete(seconds=0.05)
        assert entered.is_set()
        assert not result["qualified"] and result["state"] is None
        assert result["reason"] == "deadline"
        assert result["stationarity"] > 1e-3
        assert not s.is_current(result)
        release.set()
        result = s.run_until_complete(seconds=3)
        assert result["qualified"]
        reference(s, result)
    finally:
        release.set()
        s.close()


def test_valid_current_slow_state_does_not_wait_for_obsolete_blocked_job(monkeypatch):
    # Both free components read the same genuinely clamped observation state.
    graph = R.Graph(
        2, 3, (("input", 0, 0), ("input", 1, 1), ("state", 2, 0), ("residual", 2, 1))
    )
    s = owner(graph, clamps={2: 0.0})
    entered, release = threading.Event(), threading.Event()
    compute = s._compute

    def delayed(context):
        if context.block == 1:
            entered.set()
            assert release.wait(3)
        return compute(context)

    monkeypatch.setattr(s, "_compute", delayed)
    try:
        s.begin((0.0, 0.5), clamps={2: 0.0})
        assert entered.wait(1)
        # The old slow job is now obsolete. Current slow state is valid again;
        # subsequent fast input has no gradient dependency in the slow block.
        s.begin((0.3, 0.0), clamps={2: 0.0})
        result = s.run_until_complete(seconds=1)
        assert result["qualified"] and result["pending"] == 1
        assert not result["work_complete"]
        assert result["state"][1:] == (0.0, 0.0)
        reference(s, result)
        release.set()
        for _context, future in tuple(s._pending.values()):
            future.result(timeout=1)
        returned = s.advance()["returned"]
        assert any(r["status"] == "stale" for r in returned)
        assert s.result()["state"] == result["state"]
    finally:
        release.set()
        drained = s.close()
    assert drained["counts"]["returns"] == drained["counts"]["submissions"]
    assert drained["work_complete"]


@pytest.mark.parametrize("change", ["input", "model", "clamp"])
def test_aba_dependency_changes_veto_old_proposal(monkeypatch, change):
    s = owner()
    entered, release = threading.Event(), threading.Event()
    compute = s._compute

    def delayed(context):
        entered.set()
        assert release.wait(3)
        return compute(context)

    monkeypatch.setattr(s, "_compute", delayed)
    try:
        s.begin((0.5,))
        assert entered.wait(1)
        if change == "input":
            s.begin((0.3,), budget=0)
            s.begin((0.5,), budget=0)
        elif change == "model":
            s.begin((0.5,), weights=(0.9,), budget=0)
            s.begin((0.5,), weights=(1.0,), budget=0)
        else:
            s.begin((0.5,), clamps={0: 0.0}, budget=0)
            s.begin((0.5,), budget=0)
        release.set()
        next(iter(s._pending.values()))[1].result(timeout=1)
        rows = s.advance()["returned"]
        assert rows[0]["status"] == "stale"
        assert s._cache.state == (0.0,)
        assert not s.result()["qualified"]
    finally:
        release.set()
        s.close()


def test_budget_refuses_without_publishable_state():
    s = owner()
    try:
        s.begin((0.6,), budget=0)
        result = s.run_until_complete(seconds=1)
        assert not result["qualified"]
        assert result["state"] is result["predictions"] is result["errors"] is None
        assert result["reason"] == "budget"
        assert result["counts"]["submissions"] == 0
    finally:
        s.close()


def test_invalid_begin_is_atomic_and_all_clamped_is_explicit():
    s = owner()
    try:
        s.begin((0.2,), clamps={0: 0.4}, budget=0)
        result = s.result()
        assert result["qualified"] and result["clamps"] == ((0, 0.4),)
        reference(s, result)
        for kwargs in (
            {"clamps": {0: 2.0}},
            {"weights": (5.0,)},
            {"biases": (float("nan"),)},
        ):
            with pytest.raises(ValueError):
                s.begin((0.8,), **kwargs)
            assert s.is_current(result)
        with pytest.raises(ValueError):
            s.begin((float("nan"),))
        assert s.is_current(result)
    finally:
        s.close()


def test_forged_modified_and_closed_certificates_rejected():
    s = owner()
    s.begin((0.0,))
    result = s.result()
    assert s.is_current(result)
    assert not s.is_current({**result, "state": (0.6,)})
    assert not s.is_current(
        {"qualified": True, "context_sha256": result["context_sha256"]}
    )
    s.close()
    assert not s.is_current(result)
    assert not s.result()["qualified"]
    with pytest.raises(ValueError, match="closed"):
        s.begin((0.0,))


def test_worker_failure_and_foreign_proposal_never_publish(monkeypatch):
    s = owner()

    def broken(_context):
        raise RuntimeError("worker failed")

    monkeypatch.setattr(s, "_compute", broken)
    try:
        s.begin((0.5,))
        result = s.run_until_complete(seconds=1)
        assert not result["qualified"] and result["unknown_work"]
        assert result["counts"]["errors"] == 1
        assert result["counts"]["submissions"] == 1
    finally:
        s.close()


def test_started_wrapper_cannot_solve_if_submit_raises_after_enqueue(monkeypatch):
    s = owner()
    entered = threading.Event()
    launch, submit = s._launch, s._pool.submit
    calls, futures = [], []

    def waiting_wrapper(*args):
        entered.set()
        return launch(*args)

    def enqueued_then_failed(*args):
        futures.append(submit(*args))
        assert entered.wait(1)
        raise RuntimeError("failed after enqueue")

    monkeypatch.setattr(s, "_launch", waiting_wrapper)
    monkeypatch.setattr(s, "_compute", lambda context: calls.append(context))
    monkeypatch.setattr(s._pool, "submit", enqueued_then_failed)
    try:
        start = s.begin((0.5,))
        assert start["returned"][0]["status"] == "submit_error"
        assert futures[0].result(timeout=1).status == "cancelled_before_compute"
        assert calls == [] and s._cache.state == (0.0,)
        result = s.run_until_complete(seconds=1)
        assert result["reason"] == "submit_error"
        assert not result["qualified"] and result["state"] is None
        assert result["unknown_work"]
        assert result["counts"]["submissions"] == 1
        assert result["counts"]["submit_failures"] == 1
        assert result["counts"]["returns"] == 0
        with pytest.raises(ValueError, match="reconstruct"):
            s.begin((0.0,))
        with pytest.raises(RuntimeError, match="shutdown"):
            submit(lambda: calls.append("orphan"))
    finally:
        drained = s.close()
    assert calls == [] and drained["work_complete"] and drained["unknown_work"]


def test_thread_start_failure_cancels_queued_job_and_retires_pool(monkeypatch):
    s = owner()
    calls = []
    monkeypatch.setattr(s, "_compute", lambda context: calls.append(context))

    def failed_start(_thread):
        raise RuntimeError("thread infrastructure unavailable")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(threading.Thread, "start", failed_start)
            start = s.begin((0.4,))
        assert start["returned"][0]["status"] == "submit_error"
        with pytest.raises(RuntimeError, match="shutdown"):
            s._pool.submit(lambda: calls.append("later"))
        assert s.result()["counts"]["submit_failures"] == 1
        assert s.result()["unknown_work"]
        assert not s.result()["qualified"]
    finally:
        s.close()
    assert calls == []


def test_corrupted_worker_clamp_and_slope_rejected(monkeypatch):
    s = owner()
    compute = s._compute

    def corrupt(context):
        proposal = compute(context)
        return replace(proposal, slope=proposal.slope - 1.0)

    monkeypatch.setattr(s, "_compute", corrupt)
    try:
        s.begin((0.4,))
        result = s.run_until_complete(seconds=1)
        assert not result["qualified"]
        assert result["counts"]["errors"] == 1
        assert s._cache.state == (0.0,)
        assert result["work"]["edge_visits"] > 0
    finally:
        s.close()


def test_configuration_immutable_and_partition_complete():
    s = owner()
    try:
        with pytest.raises(AttributeError):
            s.tolerance = 100.0
        with pytest.raises(AttributeError):
            del s.blocks
    finally:
        s.close()
    with pytest.raises(ValueError, match="partition"):
        ActivitySession(R.Graph(0, 2, ()), ((0,),), (), (0.0, 0.0), (), (0.0, 0.0))


def test_qualified_fork_reuses_cache_and_has_independent_state_and_work(monkeypatch):
    import cadence._attention as module

    s = owner()
    child = None
    try:
        s.begin((0.4,))
        before = s.run_until_complete(seconds=1)

        def forbidden(*_args, **_kwargs):
            pytest.fail("fork rebuilt the numerical cache")

        monkeypatch.setattr(module, "ActivityCache", forbidden)
        child = s.fork()
        assert child._cache.state == s._cache.state
        assert child._cache is not s._cache
        assert child._pool is not s._pool
        assert child._versions is not s._versions
        assert child._counts["submissions"] == 0
        assert all(v == 0 for v in child._work.values())
        assert s.result()["work"]["forks"] > before["work"]["forks"]
        child.begin((-0.4,))
        assert child.run_until_complete(seconds=1)["qualified"]
        assert child._cache.state != s._cache.state
        assert s._cache.state == before["state"]
    finally:
        if child is not None:
            child.close()
        s.close()


def test_unqualified_and_closed_owner_cannot_fork():
    s = owner()
    s.begin((0.4,), budget=0)
    with pytest.raises(ValueError, match="qualified"):
        s.fork()
    s.close()
    with pytest.raises(ValueError, match="open"):
        s.fork()


def test_independent_jobs_really_overlap(monkeypatch):
    s = owner(R.Graph(2, 2, (("input", 0, 0), ("input", 1, 1))))
    entered = (threading.Event(), threading.Event())
    release = threading.Event()
    compute = s._compute

    def blocked(context):
        entered[context.block].set()
        assert release.wait(3)
        return compute(context)

    monkeypatch.setattr(s, "_compute", blocked)
    try:
        start = s.begin((0.2, -0.3))
        assert len(start["started"]) == 2
        assert all(event.wait(1) for event in entered)
        assert s.result()["pending"] == 2
        release.set()
        result = s.run_until_complete(seconds=2)
        assert result["qualified"]
        reference(s, result)
    finally:
        release.set()
        s.close()


def test_failed_constructor_work_is_unknown_and_context_retained(monkeypatch):
    import cadence._attention as module

    s = owner()
    s.begin((0.0,))
    before = s.result()
    cache = module.ActivityCache

    def failed(*args, **kwargs):
        cache(*args, **kwargs)  # actual work exists before the absent return
        raise RuntimeError("failure after setup")

    monkeypatch.setattr(module, "ActivityCache", failed)
    try:
        with pytest.raises(RuntimeError, match="after setup"):
            s.begin((0.4,), weights=(0.9,))
        assert s.is_current(before)
        assert s.result()["unknown_work"]
    finally:
        s.close()


@pytest.mark.parametrize("clamps", [False, [], ()])
def test_nonmapping_clamps_cannot_silently_clear_facts(clamps):
    s = owner()
    try:
        with pytest.raises(TypeError, match="mapping"):
            s.begin((0.0,), clamps=clamps)
    finally:
        s.close()


def test_line_search_refusal_is_terminal_until_context_changes(monkeypatch):
    s = owner(backtracks=3)
    monkeypatch.setattr(s, "_accept", lambda *_: False)
    try:
        s.begin((0.4,), budget=8)
        result = s.run_until_complete(seconds=2)
        assert not result["qualified"] and result["reason"] == "refused"
        assert result["counts"]["backtracks"] == 3
        assert result["counts"]["submissions"] == 1
        s.advance()
        assert s.result()["counts"]["submissions"] == 1
        assert s._cache.state == (0.0,)
    finally:
        s.close()


@pytest.mark.parametrize("seed", [31, 37, 43])
@pytest.mark.parametrize("kind", ["ordinary", "observer"])
def test_learned_twelve_patch_factual_clamps_against_public_solver(
    kind, seed, record_property
):
    c = Cortex(seed=seed)
    u = c.input("senses", shape=2)
    p = c.column("physical_relation", patches=2, inputs=u)
    h = c.column("representation", patches=4, inputs=(u, p))
    m = (
        c.observer("readback", patches=4, observes=h)
        if kind == "observer"
        else c.column("readback", patches=4, inputs=h)
    )
    f = c.column("future_relation", patches=2, inputs=(u, m))
    c.output("physical", shape=2, reads=p)
    c.output("future", shape=2, reads=f)
    brain = c.build()

    def actual(values):
        a, b = values
        return {
            "physical": (math.tanh(0.6 * a + 0.4 * b), math.tanh(-0.5 * a + 0.3 * b)),
            "future": (math.tanh(0.3 * a + 0.5 * b), math.tanh(-0.4 * a + 0.2 * b)),
        }

    rows = [
        ({"senses": (a, b)}, actual((a, b)))
        for a in (-0.4, -0.2, 0.2, 0.4)
        for b in (-0.3, 0.3)
    ]
    initial_weights = brain.weights
    for event in range(4):
        taught = brain.observe_batch(rows, event_id=event, source="witness", budget=512)
        assert taught["accepted"]
    assert brain.weights != initial_weights
    parameters = brain.weights, brain.biases, brain.inspect()["admissions"]
    s = ActivitySession.from_brain(brain, {"senses": (0.0, 0.0)})
    metrics = []
    try:
        for values in ((0.1, -0.2), (-0.3, 0.1), (0.3, 0.2), (-0.1, -0.25)):
            inputs, facts = {"senses": values}, {"physical": actual(values)["physical"]}
            expected = brain.step(inputs, targets=facts, budget=512)
            flat, fixed = brain._arguments(inputs, facts, None)
            s.begin(flat, clamps=fixed, budget=1024)
            observed = s.run_until_complete(seconds=5)
            assert expected["qualified"] and observed["qualified"]
            reference(s, observed)
            difference = max(
                abs(a - b)
                for a, b in zip(expected["state"], observed["state"], strict=True)
            )
            assert difference < 2e-5
            metrics.append(
                {
                    "state_max_difference": difference,
                    "reference_stationarity": expected["stationarity"],
                    "activity_stationarity": observed["stationarity"],
                    "reference_work": expected["work"],
                    "activity_work_cumulative": observed["work"],
                    "activity_jobs_cumulative": observed["counts"],
                }
            )
        snapshot = brain.snapshot()
        refused = brain.step({"senses": (0.9, -0.9)}, budget=0)
        assert not refused["accepted"] and brain.snapshot() == snapshot
        s.begin((0.9, -0.9), budget=0)
        refused_activity = s.run_until_complete(seconds=1)
        assert not refused_activity["qualified"] and refused_activity["state"] is None
        assert (
            brain.weights,
            brain.biases,
            brain.inspect()["admissions"],
        ) == parameters
        record_property("comparison", {"kind": kind, "seed": seed, "queries": metrics})
    finally:
        s.close()
