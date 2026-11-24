"""The bound records' compiled read (``cadence.compiled.bound_recall``) against its Python
(``BoundRecords._agreed(_taking(...))``) in every read of a life: the same fields, and their
values within rounding natively, where NumPy's BLAS sums a dot product in an order of its own;
in a browser to the bit (``kernels/wasm/browser/records.py``)."""

from __future__ import annotations

import numpy as np
import pytest

import cadence as cd
from cadence import bound, compiled

pytest.importorskip("numba")
pytestmark = pytest.mark.skipif(not compiled.available("_bound_recall"),
                                reason="the bound records' kernel is not in use")

VOICED = dict(senses={"eyes": 24, "heard": 16, "body": 4}, actions=4, code=64, bound=24,
              voice="heard", recall_capacity=8)


def test_every_read_of_a_life_is_the_python_s(monkeypatch: pytest.MonkeyPatch) -> None:
    reads = []

    def both(self, code, row):
        fast = compiled.bound_recall(self, code)
        slow = self._agreed(self._taking(code), row)
        reads.append((fast, slow))
        return fast

    monkeypatch.setattr(bound.BoundRecords, "_read", both)
    brain = cd.Brain.compose(**VOICED, backend="cpu", seed=5, arousal=True)
    rng = np.random.default_rng(9)
    words = np.eye(16)[rng.integers(0, 16, 6)]
    for i in range(120):
        x = {"eyes": rng.random((1, 24)), "body": rng.random((1, 4))}
        if i % 3 == 0:
            x["heard"] = words[i % len(words)][None]
        brain.live(x, reward=[float(rng.random() < 0.4)] if brain.pending_feedback else None)
    assert len(reads) > 100 and any(fast for fast, _ in reads)
    for fast, slow in reads:
        assert list(fast) == list(slow)
        for name in fast:
            np.testing.assert_allclose(fast[name], slow[name], rtol=0, atol=1e-9)


def test_a_cue_of_another_width_is_refused_as_the_python_refuses_it(
        monkeypatch: pytest.MonkeyPatch) -> None:
    brain = cd.Brain.compose(**VOICED, backend="cpu", seed=5, arousal=True)
    rng = np.random.default_rng(9)
    words = np.eye(16)[rng.integers(0, 16, 6)]
    for i in range(60):
        x = {"eyes": rng.random((1, 24)), "body": rng.random((1, 4))}
        if i % 3 == 0:
            x["heard"] = words[i % len(words)][None]
        brain.live(x, reward=[float(rng.random() < 0.4)] if brain.pending_feedback else None)
    records = brain.bound
    assert records.store.count
    width = records.store.keys.shape[1]
    row = np.zeros(records.inputs)
    calls = []
    kernel = compiled.bound_recall
    monkeypatch.setattr(compiled, "bound_recall", lambda *a: calls.append(1) or kernel(*a))
    records.recall(row, np.ones(width))
    assert calls == [1]  # a cue of the keys' width takes the compiled read
    for wrong in (np.ones(width // 2), np.ones(width + 8), np.ones((1, width))):
        with pytest.raises(ValueError):
            records.recall(row, wrong)
    assert calls == [1]  # and no other reaches the kernel
