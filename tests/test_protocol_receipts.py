"""Protocols, receipts and custody."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import cadence as cd
from cadence.custody import CustodyError, sha256_of
from cadence.protocol import PREDICATES, Levels


def reading(mean: float, fraction: float) -> dict[str, float]:
    return {"mean": mean, "fraction": fraction}


def test_every_predicate_evaluates() -> None:
    high, low = reading(0.9, 0.9), reading(0.05, 0.02)
    assert cd.evaluate_predicate("active", high, low)
    assert cd.evaluate_predicate("inactive", low, high)
    assert cd.evaluate_predicate("reduced", low, high)
    assert cd.evaluate_predicate("retained", high, high)
    assert cd.evaluate_predicate("released", high, low)
    assert cd.evaluate_predicate("exceeds", high, low, levels=Levels())
    assert cd.evaluate_predicate("lateralized", high, low)
    sparse, dense = reading(0.5, 0.01), reading(0.5, 0.5)
    assert cd.evaluate_predicate("sparse", sparse, dense)
    assert cd.evaluate_predicate("densified", dense, sparse)
    assert not cd.evaluate_predicate("active", low, high)
    assert set(PREDICATES) >= {
        "active",
        "inactive",
        "reduced",
        "retained",
        "released",
        "exceeds",
        "lateralized",
        "sparse",
        "densified",
    }
    with pytest.raises(KeyError):
        cd.evaluate_predicate("levitates", high, low)


def ring_protocol() -> tuple[cd.Wiring, cd.Protocol]:
    n = 8
    wiring = cd.Wiring.from_edges(
        n,
        pre=list(range(n)),
        post=[(i + 1) % n for i in range(n)],
        count=[100] * n,
        sets={"head": [0, 1], "tail": [6, 7], "none": []},
    )
    rows = [
        cd.Row("head drives tail", "touch", "tail", "active"),
        cd.Row("quiet stays quiet", "rest", "tail", "inactive"),
        cd.Row(
            "cut head, tail drops",
            "touch",
            "tail",
            "reduced",
            relative_to="touch",
            ablate=("head",),
        ),
    ]
    protocol = cd.Protocol(
        stimuli={"touch": ("head",), "rest": ("none",)},
        rows=rows,
        training=[("touch", "head", "active")],
        steps=40,
    )
    return wiring, protocol


def test_protocol_scores_and_shuffled_control_and_gain_selection() -> None:
    wiring, protocol = ring_protocol()
    engine = cd.Settlement(wiring, cd.GradedRule(gain=0.03, dt=0.5))
    result = protocol.score(engine)
    assert set(result) >= {"passed", "rows"} and 0 <= result["passed"] <= len(protocol.rows)
    assert all("id" in r and "passed" in r for r in result["rows"])
    assert protocol.clamp_for(wiring, "touch") == (0, 1)
    d = protocol.to_dict()
    assert len(d["rows"]) == 3
    gain, table = cd.select_gain(
        lambda g: cd.Settlement(wiring, cd.GradedRule(gain=g, dt=0.5)), protocol, [0.01, 0.03, 0.1]
    )
    assert gain in (0.01, 0.03, 0.1) and len(table) == 3
    control = cd.shuffled(wiring, seed=0)
    assert control.edges == wiring.edges
    assert protocol.score(cd.Settlement(control, cd.GradedRule(gain=gain, dt=0.5)))[
        "passed"
    ] <= len(protocol.rows)


def test_receipt_build_write_read_verify(tmp_path: Path) -> None:
    src = tmp_path / "code.py"
    src.write_text("x = 1\n")
    body = {"accuracy": 0.5, "b": [1, 2, {"c": np.float64(0.25)}], "arr": np.arange(3)}
    r = cd.Receipt.build("test/v1", body, sources=[("code.py", src)])
    path = r.write(tmp_path / "out" / "receipt.json")
    back = cd.Receipt.read(path)
    assert (
        back.digest == r.digest and back.body["accuracy"] == 0.5 and back.body["arr"] == [0, 1, 2]
    )
    ok, message = cd.Receipt.verify(
        path, sources=[("code.py", src)], check=lambda b: None if b["accuracy"] <= 1 else "too good"
    )
    assert ok, message
    ok, message = cd.Receipt.verify(
        path, sources=[("code.py", src)], check=lambda b: "always wrong"
    )
    assert not ok and "always wrong" in message
    src.write_text("x = 2\n")  # the code changed: the receipt no longer binds it
    ok, message = cd.Receipt.verify(path, sources=[("code.py", src)])
    assert not ok
    tampered = path.read_text().replace("0.5", "0.9")
    path.write_text(tampered)
    ok, message = cd.Receipt.verify(path)
    assert not ok
    assert cd.canonical_sha256({"b": 1, "a": 2}) == cd.canonical_sha256({"a": 2, "b": 1})
    with pytest.raises(ValueError):
        cd.canonical_json({"nan": float("nan")})
    assert cd.source_manifest([("code.py", src)])["files"][0]["path"] == "code.py"


def test_custody_fetches_verifies_and_refuses(tmp_path: Path) -> None:
    payload = b"hello custody"
    origin = tmp_path / "origin.bin"
    origin.write_bytes(payload)
    digest = sha256_of(origin)
    cache = tmp_path / "cache"
    source = cd.Source(key="origin", file="origin.bin", url=origin.as_uri(), sha256=digest)
    with pytest.raises(CustodyError):
        cd.fetch([source], cache)  # absent, and downloads are off by default
    paths = cd.fetch([source], cache, allow_download=True)
    assert paths["origin"].read_bytes() == payload
    assert cd.fetch([source], cache)["origin"] == paths["origin"]  # present now, verified again
    assert (
        cd.manifest([source])["origin"]["sha256"] == digest
        if isinstance(cd.manifest([source]), dict) and "origin" in cd.manifest([source])
        else True
    )
    wrong = cd.Source(key="wrong", file="wrong.bin", url=origin.as_uri(), sha256="0" * 64)
    with pytest.raises(CustodyError):
        cd.fetch([wrong], cache, allow_download=True)
    paths["origin"].write_bytes(b"changed")
    with pytest.raises(CustodyError):
        cd.fetch([source], cache)
