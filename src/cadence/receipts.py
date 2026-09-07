"""Receipts: results bound to the code and data that produced them.

A receipt is canonical JSON with an embedded digest, a manifest of the
source files it depends on, and whatever the experiment recorded. Verifying
a receipt recomputes the digest, re-hashes the sources, and, through a
caller-supplied check, recomputes every pass flag from the stored readings.
A receipt that cannot be verified is not a result.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["canonical_json", "canonical_sha256", "source_manifest", "Receipt"]


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def canonical_json(value: Any) -> str:
    """Compact, key-sorted JSON; NaN and infinity are refused."""
    return json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_sha256(value: Any) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def source_manifest(files: Sequence[tuple[str, Path]]) -> dict[str, Any]:
    """Digests of the source files a result depends on, keyed by their declared relative paths."""
    entries = [
        {"path": relative, "sha256": sha256(Path(path).read_bytes()).hexdigest()}
        for relative, path in files
    ]
    return {"files": entries, "manifest_sha256": canonical_sha256(entries)}


@dataclass(frozen=True)
class Receipt:
    kind: str
    body: dict[str, Any]
    source: dict[str, Any]
    digest: str

    @classmethod
    def build(
        cls, kind: str, body: Mapping[str, Any], sources: Sequence[tuple[str, Path]] = ()
    ) -> Receipt:
        source = source_manifest(sources)
        payload = {"kind": kind, "body": _plain(body), "source": source}
        return cls(kind=kind, body=payload["body"], source=source, digest=canonical_sha256(payload))

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "body": self.body, "source": self.source, "digest": self.digest}

    def write(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(canonical_json(self.to_dict()) + "\n", encoding="utf-8")
        return path

    @classmethod
    def read(cls, path: Path) -> Receipt:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(kind=raw["kind"], body=raw["body"], source=raw["source"], digest=raw["digest"])

    @classmethod
    def verify(
        cls,
        path: Path,
        *,
        sources: Sequence[tuple[str, Path]] | None = None,
        check: Callable[[dict[str, Any]], str | None] | None = None,
    ) -> tuple[bool, str]:
        """Canonical form, digest, sources, and the caller's arithmetic check must all agree."""
        raw = Path(path).read_text(encoding="utf-8")
        stored = json.loads(raw)
        if raw != canonical_json(stored) + "\n":
            return False, "receipt is not newline-terminated canonical JSON"
        payload = {k: stored[k] for k in ("kind", "body", "source")}
        if stored.get("digest") != canonical_sha256(payload):
            return False, "embedded digest does not verify"
        if sources is not None and stored["source"] != source_manifest(sources):
            return False, "source manifest differs from the current code or data"
        if check is not None:
            problem = check(stored["body"])
            if problem:
                return False, problem
        return True, "canonical form, digest, sources, and arithmetic agree"
