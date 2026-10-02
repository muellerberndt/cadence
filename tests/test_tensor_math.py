"""Independent math, actual-device and continuation checks for tensor repair."""

import json
import math
import random
from collections import Counter

import pytest

from cadence import Brain, Cortex, _repair, bootstrap
from cadence._tensor import TensorEngine

torch = pytest.importorskip("torch")

DEVICES = [
    pytest.param("cpu", "float64", id="cpu64"),
    *(
        pytest.param(
            "cuda:0",
            dtype,
            id=f"cuda{dtype.removeprefix('float')}",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA hardware unavailable"
            ),
        )
        for dtype in ("float64", "float32")
    ),
    pytest.param(
        "mps",
        "float32",
        id="mps32",
        marks=pytest.mark.skipif(
            not torch.backends.mps.is_available(), reason="MPS hardware unavailable"
        ),
    ),
]


def graph_case(seed):
    rng = random.Random(seed)
    order = (2, 0, 3, 1)
    edges = [
        ("input", 0, 2),
        ("input", 1, 0),
        ("state", 2, 0),
        ("state", 0, 2),
        ("state", 1, 1),
        ("state", 1, 3),
        ("residual", 2, 3),
        ("residual", 0, 3),
        ("residual", 3, 1),
        ("residual", 2, 1),
    ]
    rng.shuffle(edges)
    graph = _repair.Graph(2, 4, tuple(edges))
    groups = [
        [rng.uniform(-0.6, 0.6) for _ in range(size)] for size in (4, len(edges), 4)
    ]
    return graph, order, groups


def scalar_energy(graph, order, inputs, state, weights, biases, alpha, anchors, beta):
    """Explicit scalar oracle, independent of device traversal and derivatives."""
    errors = {}
    for target in order:
        drive = biases[target]
        for (kind, source, dest), weight in zip(graph.edges, weights, strict=True):
            if dest == target:
                values = (
                    inputs if kind == "input" else state if kind == "state" else errors
                )
                drive += weight * values[source]
        errors[target] = state[target] - math.tanh(drive)
    result = sum(e * e for e in errors.values()) / 2
    result += alpha * sum(x * x for x in state) / 2
    if anchors is not None:
        for values, anchor in zip((weights, biases), anchors, strict=True):
            result += (
                beta
                * sum((v - a) ** 2 for v, a in zip(values, anchor, strict=True))
                / 2
            )
    return result


@pytest.mark.parametrize(("device", "dtype"), DEVICES)
@pytest.mark.parametrize("seed", (3, 17))
@pytest.mark.parametrize("anchored", (False, True))
def test_tensor_derivatives_against_independent_scalar_energy(
    device, dtype, seed, anchored
):
    graph, order, groups = graph_case(seed)
    inputs, alpha, beta = [0.7, -0.2], 0.13, 0.29
    anchors = [[-0.12] * len(g) for g in groups[1:]] if anchored else None
    engine = TensorEngine(graph, device, dtype)
    energy, gradients = engine.evaluate(
        engine.tensor(inputs),
        *(engine.tensor(g) for g in groups),
        alpha,
        tuple(engine.tensor(g) for g in anchors) if anchored else None,
        beta,
        True,
    )
    assert energy.device.type == torch.device(device).type
    assert energy.dtype == getattr(torch, dtype)
    if device.startswith("cuda:"):
        assert energy.device == torch.device(device)
    atol = 3e-6 if dtype == "float32" else 2e-9
    expected_energy = scalar_energy(graph, order, inputs, *groups, alpha, anchors, beta)
    assert float(energy) == pytest.approx(expected_energy, rel=atol, abs=atol)
    for group_index, gradient in enumerate(gradients):
        assert gradient.device == energy.device
        assert gradient.dtype == energy.dtype
        for coordinate, analytic in enumerate(gradient.cpu().tolist()):
            plus, minus = [list(g) for g in groups], [list(g) for g in groups]
            h = 2e-6
            plus[group_index][coordinate] += h
            minus[group_index][coordinate] -= h
            upper = scalar_energy(graph, order, inputs, *plus, alpha, anchors, beta)
            lower = scalar_energy(graph, order, inputs, *minus, alpha, anchors, beta)
            assert analytic == pytest.approx(
                (upper - lower) / (2 * h), rel=atol, abs=atol
            )


@pytest.mark.parametrize(("device", "dtype"), DEVICES)
def test_tensor_query_keeps_original_frozen_parameters(device, dtype):
    graph = _repair.Graph(1, 1, (("input", 0, 0),))
    weights, biases = (0.123456789012345,), (0.234567890123456,)
    inputs = (0.345678901234567,)
    result = _repair.settle(
        graph,
        inputs,
        (0.0,),
        weights,
        biases,
        tolerance=1e-10,
        _engine=TensorEngine(graph, device, dtype),
    )
    assert result["qualified"]
    assert result["weights"] == weights
    assert result["biases"] == biases
    expected = math.tanh(weights[0] * inputs[0] + biases[0]) / 1.01
    assert result["state"] == pytest.approx((expected,), abs=1e-9)
    checked = _repair.settle(
        graph, inputs, result["state"], weights, biases, budget=0, tolerance=1e-10
    )
    assert checked["qualified"]
    assert checked["stationarity"] == result["stationarity"]


@pytest.mark.parametrize(("device", "dtype"), DEVICES)
def test_tensor_learning_uses_original_clamp_and_fixed_anchor(device, dtype):
    graph = _repair.Graph(0, 1, ())
    anchor, start, witness, prior = -0.3, 0.4, 0.1, 0.2
    result = _repair.settle(
        graph,
        (),
        (0.0,),
        (),
        (start,),
        clamps={0: witness},
        learn=True,
        tolerance=1e-10,
        anchor_weights=(),
        anchor_biases=(anchor,),
        parameter_prior=prior,
        _engine=TensorEngine(graph, device, dtype),
    )
    assert result["qualified"]
    assert result["state"] == (witness,)
    bias = result["biases"][0]
    prediction = math.tanh(bias)
    gradient = (prediction - witness) * (1 - prediction**2)
    assert abs(gradient + prior * (bias - anchor)) <= 1e-10
    assert abs(gradient + prior * (bias - start)) > 0.05
    assert result["energy"] == pytest.approx(
        (witness - prediction) ** 2 / 2
        + 0.01 * witness**2 / 2
        + prior * (bias - anchor) ** 2 / 2,
        abs=1e-15,
    )


@pytest.mark.parametrize(("device", "dtype"), DEVICES)
@pytest.mark.parametrize("learn", (False, True))
def test_nonrepresentable_bounds_are_reprojected_before_reference_check(
    device, dtype, learn
):
    graph = _repair.Graph(0, 1, ())
    result = _repair.settle(
        graph,
        (),
        (0.0,),
        (),
        (0.0 if learn else 2.0,),
        clamps={0: 0.8} if learn else {},
        learn=learn,
        state_bound=1.0 if learn else 0.1,
        parameter_bound=0.1 if learn else 4.0,
        parameter_prior=0.01,
        _engine=TensorEngine(graph, device, dtype),
    )
    assert result["qualified"]
    assert result["biases"] == (0.1,) if learn else result["state"] == (0.1,)


def test_cpu_float32_bound_regression_without_gpu():
    graph = _repair.Graph(0, 1, ())
    result = _repair.settle(
        graph,
        (),
        (0.0,),
        (),
        (2.0,),
        state_bound=0.1,
        _engine=TensorEngine(graph, "cpu", "float32"),
    )
    assert result["qualified"] and result["state"] == (0.1,)


@pytest.mark.parametrize(("device", "dtype"), DEVICES)
def test_device_stopping_hint_cannot_relax_reference_tolerance(device, dtype):
    graph = _repair.Graph(0, 1, ())
    # Binary32 stops proposing at its larger hint; reference repair must still
    # achieve the caller's much stricter numerical condition.
    engine = TensorEngine(graph, device, dtype)
    result = _repair.settle(
        graph,
        (),
        (1e-7,),
        (),
        (0.0,),
        tolerance=1e-12,
        _engine=engine,
    )
    assert result["qualified"] and result["stationarity"] <= 1e-12
    if dtype == "float32":
        assert result["execution"]["tensor_sweeps"] == 0
        assert result["execution"]["reference_sweeps"] > 0
    assert result["sweeps"] == (
        result["execution"]["tensor_sweeps"] + result["execution"]["reference_sweeps"]
    )


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA hardware unavailable"
            ),
        ),
        pytest.param(
            "mps",
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(),
                reason="MPS hardware unavailable",
            ),
        ),
    ],
)
def test_float32_device_phase_reserves_budget_for_reference_repair(monkeypatch, device):
    graph = _repair.Graph(0, 1, ())
    engine = TensorEngine(graph, device, "float32")

    def deliberately_slow_proposals(
        inputs, state, weights, biases, alpha, anchors, beta, learn
    ):
        # This proposal law keeps taking steps, even when the device budget ends.
        # The independent true objective still has to qualify within total budget.
        return -state.sum(), (
            -torch.ones_like(state),
            weights.new_empty(0),
            biases.new_empty(0),
        )

    monkeypatch.setattr(engine, "evaluate", deliberately_slow_proposals)
    result = _repair.settle(
        graph,
        (),
        (0.0,),
        (),
        (2.0,),
        budget=12,
        step=0.01,
        tolerance=1e-10,
        _engine=engine,
    )
    assert result["execution"]["tensor_sweeps"] == 6
    assert result["execution"]["reference_sweeps"] > 0
    assert result["sweeps"] <= 12
    assert result["qualified"] and result["stationarity"] <= 1e-10


@pytest.mark.parametrize(("device", "dtype"), DEVICES)
def test_nested_observers_return_influence_within_the_same_solve(device, dtype):
    cortex = Cortex(seed=7, initial_scale=0.8, device=device, dtype=dtype)
    sensor = cortex.input("sensor", shape=1)
    base = cortex.column("base", patches=2, inputs=sensor)
    middle = cortex.observer("middle", patches=2, observes=base)
    top = cortex.observer("top", patches=1, observes=(base, middle))
    cortex.output("answer", shape=1, reads=top)
    brain = cortex.build()
    snapshot = brain.snapshot()
    low = brain.settle({"sensor": [0.2]}, interventions={"top": [-0.7]})
    high = brain.settle({"sensor": [0.2]}, interventions={"top": [0.7]})
    assert low["qualified"] and high["qualified"]
    assert (
        max(
            abs(a - b) for a, b in zip(low["state"][:2], high["state"][:2], strict=True)
        )
        > 1e-3
    )
    assert brain.snapshot() == snapshot
    for result, clamp in ((low, -0.7), (high, 0.7)):
        check = _repair.settle(
            brain.graph,
            (0.2,),
            result["state"],
            brain.weights,
            brain.biases,
            clamps={4: clamp},
            budget=0,
        )
        assert check["qualified"]


@pytest.mark.parametrize(("device", "dtype"), DEVICES)
def test_work_counts_include_tensor_and_reference_evaluations(
    monkeypatch, device, dtype
):
    graph = _repair.Graph(1, 2, (("input", 0, 0), ("residual", 0, 1)))
    engine = TensorEngine(graph, device, dtype)
    counts = {"tensor": 0, "reference": 0}
    reference_edges = 0
    tensor_evaluate, reference_evaluate = engine.evaluate, _repair._evaluate

    def counted_tensor(*args, **kwargs):
        counts["tensor"] += 1
        return tensor_evaluate(*args, **kwargs)

    def counted_reference(*args, **kwargs):
        nonlocal reference_edges
        counts["reference"] += 1
        cache = kwargs.get("_query_cache")
        # Reference polishing can reuse input-only forward predictions. Every
        # reverse edge still runs; only uncached forward edges are revisited.
        reference_edges += len(graph.edges) + sum(
            len(incoming)
            for target, incoming in enumerate(graph.incoming)
            if cache is None or cache[0][target] is None
        )
        return reference_evaluate(*args, **kwargs)

    monkeypatch.setattr(engine, "evaluate", counted_tensor)
    monkeypatch.setattr(_repair, "_evaluate", counted_reference)
    result = _repair.settle(
        graph,
        (0.3,),
        (0.0, 0.0),
        (0.2, 0.4),
        (0.1, -0.1),
        tolerance=1e-10,
        _engine=engine,
    )
    assert result["qualified"]
    assert result["work"]["evaluations"] == sum(counts.values())
    assert result["work"]["patch_visits"] == 2 * graph.n_patches * sum(counts.values())
    assert result["work"]["edge_visits"] == (
        2 * len(graph.edges) * counts["tensor"] + reference_edges
    )
    assert result["execution"]["reference_evaluations"] == counts["reference"]
    assert (
        result["work"]["proposals"] == result["sweeps"] + result["work"]["backtracks"]
    )


def test_reference_check_rejects_device_candidate_with_higher_true_energy(monkeypatch):
    graph = _repair.Graph(0, 1, ())
    engine = TensorEngine(graph, "cpu", "float32")

    def faulty_proposals(inputs, state, weights, biases, alpha, anchors, beta, learn):
        # Deliberately wrong tensor objective: the independent reference must stop it.
        delta = state - 0.5
        return delta.square().sum(), (
            2 * delta,
            weights.new_empty(0),
            biases.new_empty(0),
        )

    monkeypatch.setattr(engine, "evaluate", faulty_proposals)
    result = _repair.settle(graph, (), (0.1,), (), (0.0,), budget=8, _engine=engine)
    assert result["qualified"]
    assert result["state"] == pytest.approx((0.0,), abs=1e-6)
    assert result["execution"]["reference_restart"]
    assert result["energy"] < 1e-12


def make_brain(device="python", dtype=None):
    cortex = Cortex(seed=7, device=device, dtype=dtype)
    sensor = cortex.input("sensor", shape=1)
    base = cortex.column("base", patches=2, inputs=sensor)
    observer = cortex.observer("observer", patches=1, inputs=sensor, observes=base)
    cortex.output("answer", shape=1, reads=observer)
    return cortex.build()


@pytest.mark.parametrize(("device", "dtype"), DEVICES)
def test_explicit_checkpoint_transfer_preserves_values_and_event_custody(device, dtype):
    original = make_brain()
    inputs, targets = {"sensor": [0.3]}, {"answer": [0.1]}
    assert original.observe(inputs, targets, event_id=4)["accepted"]
    snapshot = original.snapshot()
    transferred = Brain.from_snapshot(snapshot, device=device, dtype=dtype)
    assert transferred.config["device"] == device
    assert transferred.config["dtype"] == dtype
    assert transferred.weights == original.weights
    assert transferred.biases == original.biases
    assert transferred.state == original.state
    assert transferred.inspect()["fingerprint"] != original.inspect()["fingerprint"]
    before = transferred.snapshot()
    assert transferred.observe(inputs, targets, event_id=4)["duplicate"]
    assert transferred.snapshot() == before
    result = transferred.step(inputs)
    assert result["accepted"]
    assert result["execution"]["device"] == device
    assert transferred.weights == original.weights
    assert transferred.biases == original.biases
    restored = Brain.from_snapshot(transferred.snapshot())
    assert restored.snapshot() == transferred.snapshot()


@pytest.mark.parametrize("tamper", ("source", "config", "fingerprint"))
def test_checkpoint_overrides_do_not_bypass_original_identity_validation(tamper):
    data = json.loads(make_brain().snapshot())
    if tamper == "source":
        assert "_tensor.py" in data["implementation"]
        data["implementation"]["_tensor.py"] = "0" * 64
    elif tamper == "config":
        data["config"]["device"] = "cpu"
    else:
        data["fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="mismatch"):
        Brain.from_snapshot(json.dumps(data), device="cpu", dtype="float64")


@pytest.mark.parametrize(("device", "dtype"), DEVICES)
def test_refused_tensor_admission_does_not_consume_event(device, dtype):
    brain = make_brain(device, dtype)
    before = brain.snapshot()
    inputs, targets = {"sensor": [0.3]}, {"answer": [0.8]}
    result = brain.observe(inputs, targets, event_id=5, budget=1)
    assert not result["accepted"] and result["reason"] == "budget"
    assert result["execution"]["tensor_sweeps"] == 1
    assert brain.snapshot() == before
    result = brain.observe(inputs, targets, event_id=5)
    assert result["accepted"]
    assert brain.inspect()["last_event_id"] == 5


@pytest.mark.parametrize(("device", "dtype"), DEVICES)
def test_tensor_pre_activation_overflow_is_rejected(device, dtype):
    graph = _repair.Graph(1, 1, (("input", 0, 0),))
    engine = TensorEngine(graph, device, dtype)
    big = 1e308 if dtype == "float64" else 1e38
    with pytest.raises(ValueError, match="prediction.*finite numeric range"):
        engine.evaluate(
            engine.tensor((big,)),
            engine.tensor((0.0,)),
            engine.tensor((4.0,)),
            engine.tensor((0.0,)),
            0.01,
            None,
            0.1,
            False,
        )


@pytest.mark.parametrize(("device", "dtype"), DEVICES)
def test_query_skips_unused_parameter_gradients_on_device(device, dtype):
    graph = _repair.Graph(1, 1, (("input", 0, 0),))
    engine = TensorEngine(graph, device, dtype)
    big = 1e308 if dtype == "float64" else 1e38
    args = (
        engine.tensor((big,)),
        engine.tensor((4.0,)),
        engine.tensor((0.0,)),
        engine.tensor((0.0,)),
        0.01,
        None,
        0.1,
    )
    energy, gradients = engine.evaluate(*args, False)
    assert math.isfinite(float(energy))
    assert gradients[1].numel() == gradients[2].numel() == 0
    with pytest.raises(ValueError, match="gradient exceeds"):
        engine.evaluate(*args, True)


@pytest.mark.parametrize(("device", "dtype"), DEVICES)
def test_bootstrap_held_out_recall_and_live_learning_on_device(
    monkeypatch, device, dtype
):
    cortex = Cortex(seed=2, device=device, dtype=dtype)
    sensor = cortex.input("signal", shape=1)
    base = cortex.column("base", patches=4, inputs=sensor)
    reflection = cortex.observer("reflection", patches=2, inputs=sensor, observes=base)
    cortex.output("answer", shape=1, reads=reflection)
    brain = cortex.build()
    examples = [({"signal": [v]}, {"answer": [v]}) for v in (-0.8, 0.8)]
    checks = [({"signal": [v]}, {"answer": [v]}) for v in (-0.4, 0.4)]
    solves = []
    original_solve = brain._solve

    def recorded_solve(inputs, clamps, *, learn, budget):
        result = original_solve(inputs, clamps, learn=learn, budget=budget)
        solves.append((inputs, learn, result))
        return result

    monkeypatch.setattr(brain, "_solve", recorded_solve)
    report = bootstrap(brain, examples, checks=checks, max_error=0.2, seed=11)
    assert report["passed"] and report["failure"] is None
    assert report["accepted"] == report["presentations"] == 2 * report["epochs"]
    assert report["accepted"] > 0
    assert len(solves) == 4 * len(report["history"]) + report["accepted"]
    work = Counter()
    for inputs, learn, result in solves:
        assert (
            result["qualified"] and result["stationarity"] <= brain.config["tolerance"]
        )
        assert result["execution"]["device"] == device
        work.update(result["work"])
        if learn:
            assert inputs[0] in (-0.8, 0.8)
    assert dict(work) == report["work"]
    assert (
        sum(
            result["execution"]["tensor_sweeps"] for _, learn, result in solves if learn
        )
        > 0
    )
    for entry in report["history"]:
        assert entry["recall"]["qualified"] == entry["recall"]["evaluated"] == 2
        assert entry["checks"]["qualified"] == entry["checks"]["evaluated"] == 2
    assert brain.inspect()["admissions"] == report["accepted"]
    saved = brain.snapshot()
    # These amplitudes were absent from both teaching and readiness queries.
    for value in (-0.6, 0.6):
        assert abs(brain.predict({"signal": [value]})["answer"][0] - value) < 0.2
    assert brain.snapshot() == saved
    live = Brain.from_snapshot(saved)
    inputs, targets = {"signal": [0.2]}, {"answer": [-0.3]}
    old_prediction = live.predict(inputs)["answer"][0]
    admitted = live.observe(inputs, targets)
    assert admitted["accepted"]
    assert live.inspect()["admissions"] == report["accepted"] + 1
    assert abs(live.predict(inputs)["answer"][0] + 0.3) < abs(old_prediction + 0.3)
    assert live.weights != brain.weights
    before_retry = live.snapshot()
    assert live.observe(inputs, targets, event_id=admitted["event_id"])["duplicate"]
    assert live.snapshot() == before_retry
    assert brain.snapshot() == saved
