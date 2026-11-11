from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import cadence as cd
from cadence.receipts import canonical_sha256


def ring(n: int = 6, count: float = 120.0) -> cd.Connectome:
    pre = list(range(n))
    post = [(i + 1) % n for i in range(n)]
    return cd.Connectome.from_synapses(
        n, pre=pre, post=post, count=[count] * n, populations={"head": [0], "tail": [n - 1]}
    )


def test_connectome_sorts_merges_and_drops_autapses() -> None:
    w = cd.Connectome.from_synapses(
        3, pre=[1, 0, 0, 2], post=[0, 1, 1, 2], count=[5, 3, 4, 9], sign=[1, 1, -1, 1]
    )
    assert w.synapses == 2  # autapse 2->2 dropped, parallel 0->1 merged
    assert w.post.tolist() == [0, 1] and w.pre.tolist() == [1, 0]
    assert w.count.tolist() == [5.0, 7.0]
    assert w.sign[1] == pytest.approx((3 - 4) / 7)


def test_rest_is_a_fixed_point_and_a_clamp_propagates() -> None:
    brain = cd.NeuralGraph(ring(), cd.NeuronModel(gain=0.03))
    assert brain.settle(None, steps=20).activation.max() == 0.0
    state = brain.settle({0: 1.0}, steps=80)
    assert state.activation.min() > 0.9  # the ring lights up all the way round


def test_ablation_silences_downstream() -> None:
    w = ring()
    brain = cd.NeuralGraph(w, cd.NeuronModel(gain=0.03))
    mask = np.ones(w.n)
    mask[2] = 0.0
    state = brain.settle([0], steps=80, mask=mask)
    assert state.activation[1] > 0.9 and state.activation[2] == 0.0 and state.activation[3] < 0.05


def test_adaptation_turns_a_half_center_into_a_rhythm() -> None:
    # Two neurons, each excited by the stimulus and inhibiting the other: a half-center oscillator.
    w = cd.Connectome.from_synapses(2, pre=[0, 1], post=[1, 0], count=[60, 60], sign=[-1, -1])
    still = cd.NeuralGraph(w, cd.NeuronModel(gain=0.03))
    fixed = still.settle({0: 1.0, 1: 0.95}, steps=400, trajectory=True)
    assert fixed.trajectory is not None and fixed.trajectory[-100:].std(axis=0).max() < 1e-3
    rhythmic = cd.NeuralGraph(
        w, cd.NeuronModel(gain=0.03, adaptation=cd.Adaptation(tau_steps=40, strength=2.0))
    )
    moving = rhythmic.settle({0: 1.0, 1: 0.95}, steps=400, trajectory=True)
    assert moving.trajectory is not None and moving.trajectory[-200:].std(axis=0).max() > 0.2


def test_cpu_backend_matches_neuron_by_neuron_reference() -> None:
    rng = np.random.default_rng(1)
    n, e = 40, 300
    w = cd.Connectome.from_synapses(
        n,
        pre=rng.integers(0, n, e),
        post=rng.integers(0, n, e),
        count=rng.integers(5, 60, e),
        sign=np.where(rng.random(e) < 0.3, -1.0, 1.0),
    )
    neuron_model = cd.NeuronModel(gain=0.02, adaptation=cd.Adaptation(tau_steps=25, strength=0.5))
    brain = cd.NeuralGraph(
        w, neuron_model, log_gain=0.1 * rng.standard_normal(n), bias=0.05 * rng.standard_normal(n)
    )
    report = cd.conformance(brain, list(range(5)), steps=40)
    assert report["ledger"]["clean"] and report["max_abs_deviation"] < 1e-12


@pytest.mark.skipif("torch" not in cd.available_backends(), reason="torch not installed")
def test_torch_backend_agrees_with_cpu_within_its_precision() -> None:
    rng = np.random.default_rng(2)
    n, e = 300, 3000
    w = cd.Connectome.from_synapses(
        n, pre=rng.integers(0, n, e), post=rng.integers(0, n, e), count=rng.integers(5, 40, e)
    )
    neuron_model = cd.NeuronModel(gain=0.02)
    cpu = cd.NeuralGraph(w, neuron_model).settle(list(range(10)), steps=30, trajectory=True)
    acc = cd.NeuralGraph(w, neuron_model, backend="torch").settle(
        list(range(10)), steps=30, trajectory=True
    )
    assert cpu.trajectory is not None and acc.trajectory is not None
    tolerance = 1e-3 if "float32" in cd.available_backends()["torch"] else 1e-9
    assert np.abs(cpu.trajectory - acc.trajectory).max() < tolerance


def test_receipt_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    src = tmp_path / "code.py"
    src.write_text("print('hi')\n")
    receipt = cd.Receipt.build(
        "demo/v1", {"passed": 3, "rows": [{"id": "R1", "passed": True}]}, [("code.py", src)]
    )
    path = receipt.write(tmp_path / "receipt.json")
    ok, message = cd.Receipt.verify(
        path, sources=[("code.py", src)], check=lambda b: None if b["passed"] == 3 else "tally"
    )
    assert ok, message
    raw = json.loads(path.read_text())
    raw["body"]["passed"] = 4
    raw["digest"] = canonical_sha256({k: raw[k] for k in ("kind", "body", "source")})
    path.write_text(cd.canonical_json(raw) + "\n")
    ok, message = cd.Receipt.verify(
        path, check=lambda b: None if b["passed"] == 3 else "tally does not match rows"
    )
    assert not ok and "tally" in message
    src.write_text("print('changed')\n")
    ok, message = cd.Receipt.verify(path, sources=[("code.py", src)])
    assert not ok and "source" in message
