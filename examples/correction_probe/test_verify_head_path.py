"""Independent no-model-call checks of saved-state arithmetic and the census."""

import copy
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import body
import head_path as H
import probe as P
import verify_head_path as V


@pytest.fixture(autouse=True)
def no_calls(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No numerical model calls")

    for name in ("settle", "step", "observe", "observe_batch", "predict"):
        monkeypatch.setattr(P.Brain, name, forbidden)


def test_exact_chronology_and_factual_targets():
    tape = body.data()
    calls = list(V.expected(tape, P.schedule(3109)))
    assert len(calls) == 1428
    assert Counter(c[0] for c in calls) == {"settle": 1296, "observe_batch": 132}
    assert Counter(c[3]["phase"] for c in calls)["training"] == 128
    for method, argument, targets, meta, row in calls:
        if method == "settle":
            if "f" in (targets or {}):
                assert meta["phase"] == "teacher_clamp_diagnostic" and meta["teacher"]
            assert argument == body.inputs(row)
        else:
            assert len(argument) == 16 and targets is None
            for _, labels in argument:
                assert set(labels) in ({"p0", "c", "f"}, {"p1", "c", "f"})


@pytest.mark.parametrize(
    "kind",
    ["head_ordinary", "head_observer", "head_wide", "raw_ordinary", "raw_observer"],
)
def test_analytic_zero_equilibrium_and_forged_prediction_rejected(kind):
    brain = H.make(kind, 3109)
    brain._weights = tuple(0.0 for _ in brain.weights)
    brain._biases = tuple(0.0 for _ in brain.biases)
    n = brain.inspect()["patches"]
    inputs = {
        k: [0.0] * size
        for k, size in (
            ("coarse", 2),
            ("command", 2),
            ("prior", 3),
            ("fine", 3),
            ("present", 3),
        )
    }
    result = {
        "state": [0.0] * n,
        "predictions": [0.0] * n,
        "errors": [0.0] * n,
        "energy": 0.0,
        "stationarity": 0.0,
        "outputs": {k: [0.0] for k in ("p0", "p1", "c", "f")},
        "parameter_sha256": P.digest([brain.weights, brain.biases]),
    }
    assert V.qualified_state(brain, result, inputs, None) == 0
    altered = copy.deepcopy(result)
    altered["predictions"][-1] = 0.1
    with pytest.raises(AssertionError):
        V.qualified_state(brain, altered, inputs, None)
    with pytest.raises(AssertionError):
        V.qualified_state(brain, result, inputs, {"c": [0.1]})


def test_invalid_state_domain_rejected_before_projection():
    brain = H.make("head_observer", 3109)
    n = len(brain.biases)
    inputs = {
        k: [0.0] * size
        for k, size in (
            ("coarse", 2),
            ("command", 2),
            ("prior", 3),
            ("fine", 3),
            ("present", 3),
        )
    }
    for values in (
        [0.0] * (n + 1),
        [float("nan")] * n,
        [brain.config["state_bound"] + 1] * n,
    ):
        with pytest.raises((AssertionError, ValueError)):
            V.qualified_state(brain, {"state": values}, inputs, None)
