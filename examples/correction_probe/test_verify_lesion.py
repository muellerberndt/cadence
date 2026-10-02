"""Independent arithmetic checks; no settling, fitting or body execution."""

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe as P
from verify_lesion import arithmetic


@pytest.fixture(autouse=True)
def no_solver(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No solver calls in arithmetic tests")

    for name in ("settle", "step", "observe", "observe_batch", "predict"):
        monkeypatch.setattr(P.Brain, name, forbidden)


def test_two_patch_return_derivative_closed_form():
    graph = SimpleNamespace(
        residual_order=(0, 1), incoming=((), (0,)), edges=(("residual", 0, 1),)
    )
    x, y, w, alpha = 0.2, -0.1, 0.7, 0.01
    p, e, g, energy = arithmetic(graph, (), (x, y), (w,), (0.0, 0.0), alpha)
    q = math.tanh(w * x)
    assert p == [0.0, q] and e == [x, y - q]
    assert g == pytest.approx(
        [(1 + alpha) * x - w * (y - q) * (1 - q * q), (1 + alpha) * y - q], abs=1e-15
    )
    assert energy == pytest.approx(
        0.5 * (x * x + (y - q) ** 2) + 0.5 * alpha * (x * x + y * y), abs=1e-15
    )


@pytest.mark.parametrize("kind", P.KINDS)
def test_full_layout_scalar_gradient_matches_finite_difference(kind):
    brain = P.make(kind, 3109)
    inputs = [0.4 * math.sin(i + 1) for i in range(13)]
    state = [0.3 * math.cos(i + 1) for i in range(brain.graph.n_patches)]
    args = (brain.graph, inputs)
    rest = (brain.weights, brain.biases, brain.config["state_prior"])
    grad = arithmetic(*args, state, *rest)[2]
    for i in range(len(state)):
        high, low = list(state), list(state)
        high[i] += 1e-6
        low[i] -= 1e-6
        difference = (
            arithmetic(*args, high, *rest)[3] - arithmetic(*args, low, *rest)[3]
        ) / 2e-6
        assert grad[i] == pytest.approx(difference, abs=2e-10)
