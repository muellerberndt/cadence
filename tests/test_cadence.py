from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import cadence as cd


def ring(n: int = 6, count: float = 120.0) -> cd.Wiring:
    pre = list(range(n))
    post = [(i + 1) % n for i in range(n)]
    return cd.Wiring.from_edges(
        n, pre=pre, post=post, count=[count] * n, sets={"head": [0], "tail": [n - 1]}
    )


def test_wiring_sorts_merges_and_drops_autapses() -> None:
    w = cd.Wiring.from_edges(
        3, pre=[1, 0, 0, 2], post=[0, 1, 1, 2], count=[5, 3, 4, 9], sign=[1, 1, -1, 1]
    )
    assert w.edges == 2  # autapse 2->2 dropped, parallel 0->1 merged
    assert w.post.tolist() == [0, 1] and w.pre.tolist() == [1, 0]
    assert w.count.tolist() == [5.0, 7.0]
    assert w.sign[1] == pytest.approx((3 - 4) / 7)


def test_rest_is_a_fixed_point_and_a_clamp_propagates() -> None:
    engine = cd.Settlement(ring(), cd.GradedRule(gain=0.03))
    assert engine.settle(None, steps=20).activation.max() == 0.0
    state = engine.settle({0: 1.0}, steps=80)
    assert state.activation.min() > 0.9  # the ring lights up all the way round


def test_ablation_silences_downstream() -> None:
    w = ring()
    engine = cd.Settlement(w, cd.GradedRule(gain=0.03))
    mask = np.ones(w.n)
    mask[2] = 0.0
    state = engine.settle([0], steps=80, mask=mask)
    assert state.activation[1] > 0.9 and state.activation[2] == 0.0 and state.activation[3] < 0.05


def test_adaptation_turns_a_half_center_into_a_rhythm() -> None:
    # Two owners, each excited by the clamp and inhibiting the other: a half-center oscillator.
    w = cd.Wiring.from_edges(2, pre=[0, 1], post=[1, 0], count=[60, 60], sign=[-1, -1])
    still = cd.Settlement(w, cd.GradedRule(gain=0.03))
    fixed = still.settle({0: 1.0, 1: 0.95}, steps=400, trajectory=True)
    assert fixed.trajectory is not None and fixed.trajectory[-100:].std(axis=0).max() < 1e-3
    rhythmic = cd.Settlement(
        w, cd.GradedRule(gain=0.03, adaptation=cd.Adaptation(tau_steps=40, strength=2.0))
    )
    moving = rhythmic.settle({0: 1.0, 1: 0.95}, steps=400, trajectory=True)
    assert moving.trajectory is not None and moving.trajectory[-200:].std(axis=0).max() > 0.2


def test_cpu_backend_matches_owner_by_owner_reference() -> None:
    rng = np.random.default_rng(1)
    n, e = 40, 300
    w = cd.Wiring.from_edges(
        n,
        pre=rng.integers(0, n, e),
        post=rng.integers(0, n, e),
        count=rng.integers(5, 60, e),
        sign=np.where(rng.random(e) < 0.3, -1.0, 1.0),
    )
    rule = cd.GradedRule(gain=0.02, adaptation=cd.Adaptation(tau_steps=25, strength=0.5))
    engine = cd.Settlement(
        w, rule, log_gain=0.1 * rng.standard_normal(n), bias=0.05 * rng.standard_normal(n)
    )
    report = cd.conformance(engine, list(range(5)), steps=40)
    assert report["ledger"]["clean"] and report["max_abs_deviation"] < 1e-12


@pytest.mark.skipif("torch" not in cd.available_backends(), reason="torch not installed")
def test_torch_backend_agrees_with_cpu_within_its_precision() -> None:
    rng = np.random.default_rng(2)
    n, e = 300, 3000
    w = cd.Wiring.from_edges(
        n, pre=rng.integers(0, n, e), post=rng.integers(0, n, e), count=rng.integers(5, 40, e)
    )
    rule = cd.GradedRule(gain=0.02)
    cpu = cd.Settlement(w, rule).settle(list(range(10)), steps=30, trajectory=True)
    acc = cd.Settlement(w, rule, backend="torch").settle(list(range(10)), steps=30, trajectory=True)
    assert cpu.trajectory is not None and acc.trajectory is not None
    tolerance = 1e-3 if "float32" in cd.available_backends()["torch"] else 1e-9
    assert np.abs(cpu.trajectory - acc.trajectory).max() < tolerance


def test_protocol_scores_and_shuffle_keeps_counts() -> None:
    w = ring(8)
    protocol = cd.Protocol(
        stimuli={"baseline": (), "poke": ("head",)},
        training=[("poke", "tail", "active")],
        rows=[
            cd.Row("R1", "baseline", "tail", "inactive", "nothing in, nothing out"),
            cd.Row("R2", "poke", "tail", "reduced", "cutting the ring", ablate=("head",)),
        ],
        steps=80,
    )
    engine = cd.Settlement(w, cd.GradedRule(gain=0.03))
    report = protocol.score(engine)
    assert report["training_passed"] == 1 and report["passed"] == 2
    control = cd.shuffled(w, seed=0)
    assert (
        control.count.sum() == w.count.sum()
        and control.out_degree().tolist() == w.out_degree().tolist()
    )
    assert control.sets == w.sets
    gain, table = cd.select_gain(
        lambda g: cd.Settlement(w, cd.GradedRule(gain=g)),
        protocol,
        (0.005, 0.03, 0.2),
        sparsity_cap=None,  # a toy ring lights entirely; the cap is for large nets
    )
    assert gain == 0.03 and [row["admissible"] for row in table] == [True, True, True]


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
    raw["digest"] = cd.canonical_sha256({k: raw[k] for k in ("kind", "body", "source")})
    path.write_text(cd.canonical_json(raw) + "\n")
    ok, message = cd.Receipt.verify(
        path, check=lambda b: None if b["passed"] == 3 else "tally does not match rows"
    )
    assert not ok and "tally" in message
    src.write_text("print('changed')\n")
    ok, message = cd.Receipt.verify(path, sources=[("code.py", src)])
    assert not ok and "source" in message


def test_custody_refuses_unpinned_files(tmp_path: Path) -> None:
    data = tmp_path / "wiring.csv"
    data.write_text("a,b\n")
    good = cd.Source(
        "wiring", "wiring.csv", "https://example.invalid/wiring.csv", cd.custody.sha256_of(data)
    )
    assert cd.fetch([good], tmp_path)["wiring"] == data
    bad = cd.Source("wiring", "wiring.csv", "https://example.invalid/wiring.csv", "0" * 64)
    with pytest.raises(cd.custody.CustodyError):
        cd.fetch([bad], tmp_path)
    with pytest.raises(cd.custody.CustodyError):
        cd.fetch([cd.Source("x", "missing.csv", "https://example.invalid/x", "0" * 64)], tmp_path)
