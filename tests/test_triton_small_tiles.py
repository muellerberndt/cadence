"""The torch backend's small-group Triton kernel at every tile shape (cadence#299): where the
installed Triton refuses ``tl.dot`` below 16 on a side, the kernel sums the same products
without it, and the settled states equal the cuBLAS path's (``CADENCE_TRITON=0``) within the
precision's rounding. Runs where CUDA and Triton exist."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available() or importlib.util.find_spec("triton") is None:
    pytest.skip("CUDA and Triton are needed", allow_module_level=True)

from cadence import Brain  # noqa: E402
from cadence import brain as graph  # noqa: E402

TOLERANCE = {"float32": 1e-4, "float64": 1e-12}  # a lesson and a settle in that precision

def _host(value):
    """An array on the host, from a tensor or an array."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)



def _settled(options: dict, precision: str, rows: int, triton: bool, monkeypatch) -> tuple:
    monkeypatch.setenv("CADENCE_TRITON", "1" if triton else "0")
    brain = Brain.compose(576, 3, seed=0, backend="torch", device="cuda", precision=precision,
                          **options)
    rng = np.random.default_rng(0)
    x = rng.random((rows, 576))
    brain.lesson(x, rng.integers(3, size=rows))
    actions = np.asarray(brain.act(x, greedy=True))
    state = brain._qualified(brain.stimulus(x, memory=False), operation="predict")
    return actions, _host(state.activation)


@pytest.mark.parametrize("options", [{"recall": False}, {}], ids=["recall-off", "composed"])
@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("rows", [1, 8, 16, 32])
def test_the_kernel_settles_as_the_cublas_path(options, precision, rows, monkeypatch):
    actions, state = _settled(options, precision, rows, True, monkeypatch)
    reference_actions, reference = _settled(options, precision, rows, False, monkeypatch)
    np.testing.assert_array_equal(actions, reference_actions)
    np.testing.assert_allclose(state, reference, atol=TOLERANCE[precision], rtol=0)
    assert graph._DOT_SHAPES  # the launch ran and chose a product per tile shape
