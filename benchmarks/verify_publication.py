"""Verify the bytes and envelope digests of the published benchmark copies."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path(__file__).with_name("publication-manifest.json")


def canonical_digest(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def verify(root: Path = ROOT, manifest: Path = MANIFEST) -> int:
    ledger = json.loads(manifest.read_text())
    assert ledger["format"] == "cadence-publication-normalization/1"
    names: set[str] = set()
    for item in ledger["files"]:
        name = item["file"]
        assert name not in names, f"duplicate publication entry: {name}"
        names.add(name)
        path = root / name
        assert path.resolve().is_relative_to(root.resolve()), name
        data = path.read_bytes()
        assert len(data) == item["file_bytes"], f"published size differs: {name}"
        assert hashlib.sha256(data).hexdigest() == item["file_sha256"], name
        raw = gzip.decompress(data) if item["gzip"] else data
        assert len(raw) == item["raw_bytes"], f"raw size differs: {name}"
        assert hashlib.sha256(raw).hexdigest() == item["raw_sha256"], name
        if "published_receipt_digest" in item:
            receipt = json.loads(raw)
            payload = (
                receipt["body"]
                if item["receipt_digest_scope"] == "body"
                else {k: v for k, v in receipt.items() if k != "digest"}
            )
            assert canonical_digest(payload) == receipt["digest"], name
            assert receipt["digest"] == item["published_receipt_digest"], name
    return len(names)


if __name__ == "__main__":
    print(f"Verified {verify()} published benchmark copies and their receipt digests.")
