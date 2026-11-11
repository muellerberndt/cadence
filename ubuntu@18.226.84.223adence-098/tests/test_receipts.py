"""Direct tests for cadence.receipts: Receipt build/write/read/verify and canonical helpers."""

import json
from pathlib import Path

from cadence.receipts import Receipt, canonical_json, canonical_sha256

# ------------------------------------------------------------------ canonical helpers


def test_canonical_json_is_sorted_and_compact() -> None:
    obj = {"b": 2, "a": 1}
    out = canonical_json(obj)
    parsed = json.loads(out)
    assert list(parsed.keys()) == ["a", "b"]


def test_canonical_sha256_is_64_hex_chars() -> None:
    digest = canonical_sha256({"key": "value"})
    assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)


def test_canonical_sha256_is_deterministic() -> None:
    obj = {"x": [1, 2, 3]}
    assert canonical_sha256(obj) == canonical_sha256(obj)


def test_canonical_sha256_differs_for_different_inputs() -> None:
    assert canonical_sha256({"a": 1}) != canonical_sha256({"a": 2})


# ------------------------------------------------------------------ Receipt.build


def test_receipt_build_populates_all_fields() -> None:
    r = Receipt.build("test_run", {"accuracy": 0.95})
    assert r.kind == "test_run"
    assert r.body["accuracy"] == 0.95
    assert len(r.digest) == 64


def test_receipt_digest_is_stable_across_builds() -> None:
    r1 = Receipt.build("run", {"n": 42})
    r2 = Receipt.build("run", {"n": 42})
    assert r1.digest == r2.digest


def test_receipt_different_body_gives_different_digest() -> None:
    r1 = Receipt.build("run", {"n": 1})
    r2 = Receipt.build("run", {"n": 2})
    assert r1.digest != r2.digest


# ------------------------------------------------------------------ Receipt.write / read / verify


def test_receipt_write_read_roundtrip(tmp_path: Path) -> None:
    r = Receipt.build("experiment", {"seed": 7, "loss": 0.42})
    p = r.write(tmp_path / "result.json")
    loaded = Receipt.read(p)
    assert loaded.kind == r.kind
    assert loaded.digest == r.digest
    assert loaded.body == r.body


def test_receipt_verify_passes_for_valid_file(tmp_path: Path) -> None:
    r = Receipt.build("check", {"val": True})
    p = r.write(tmp_path / "check.json")
    ok, msg = Receipt.verify(p)
    assert ok, msg


def test_receipt_verify_fails_for_tampered_digest(tmp_path: Path) -> None:
    r = Receipt.build("check", {"val": 1})
    p = r.write(tmp_path / "tampered.json")
    raw = json.loads(p.read_text())
    raw["digest"] = "0" * 64
    p.write_text(canonical_json(raw) + "\n")
    ok, msg = Receipt.verify(p)
    assert not ok
    assert "digest" in msg


def test_receipt_to_dict_has_four_fields() -> None:
    r = Receipt.build("t", {"x": 1})
    d = r.to_dict()
    assert set(d.keys()) == {"kind", "body", "source", "digest"}
