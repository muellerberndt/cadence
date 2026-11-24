"""The backends a test of the same contract on every backend reads it on: the NumPy host, a torch
device in each precision it has, and MLX. ``place`` gives one's options, or skips where it is not
here; ``PLACES`` names them for ``pytest.mark.parametrize``."""

from __future__ import annotations

from typing import Any

import pytest

import cadence as cd

_PLACES: dict[str, dict[str, Any]] = {
    "numpy": {"backend": "cpu"},
    "torch-cpu-float64": {"backend": "torch", "device": "cpu", "precision": "float64"},
    "torch-cpu-float32": {"backend": "torch", "device": "cpu", "precision": "float32"},
    "torch-mps": {"backend": "torch", "device": "mps", "precision": "float32"},
    "torch-cuda-float64": {"backend": "torch", "device": "cuda", "precision": "float64"},
    "torch-cuda-float32": {"backend": "torch", "device": "cuda", "precision": "float32"},
    "mlx": {"backend": "mlx"},
}
PLACES = list(_PLACES)
DEVICES = [name for name in PLACES if name != "numpy"]


def place(name: str) -> dict[str, Any]:
    """The options that put a graph or a brain on the backend ``name``."""
    where = _PLACES[name]
    if where["backend"] == "mlx" and "mlx" not in cd.available_backends():
        pytest.skip("mlx not installed")
    if where["backend"] == "torch":
        torch = pytest.importorskip("torch")
        if where["device"] == "mps" and not torch.backends.mps.is_available():
            pytest.skip("MPS unavailable")
        if where["device"] == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
    return dict(where)


def rounding(name: str) -> float:
    """What two readings of one number may differ by on ``name``: nothing in float64, float32's
    rounding at activities near one elsewhere."""
    return 0.0 if name == "numpy" or name.endswith("float64") else 1e-6
