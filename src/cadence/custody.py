"""Custody: pinned public sources, fetched once, verified every time.

A lane that reads measured wiring declares each source file by URL and
SHA-256. Nothing downloads unless asked; a download lands in a temporary
file and is renamed only after its digest matches, so a corrupt payload
never becomes the cached source. The manifest of the sources is what a
receipt embeds.
"""

from __future__ import annotations

import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

__all__ = ["Source", "fetch", "manifest", "sha256_of", "CustodyError"]


class CustodyError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Source:
    key: str
    file: str
    url: str
    sha256: str
    citation: str = ""


def sha256_of(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(
    sources: Sequence[Source], root: Path, *, allow_download: bool = False
) -> dict[str, Path]:
    """Verified local paths for every source; downloads only when ``allow_download`` is set."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for source in sources:
        target = root / source.file
        if not target.exists():
            if not allow_download:
                raise CustodyError(
                    f"{target} is absent; pass allow_download=True to fetch {source.url}"
                )
            temporary = target.with_suffix(target.suffix + ".part")
            with urllib.request.urlopen(source.url, timeout=600) as response:
                temporary.write_bytes(response.read())
            if sha256_of(temporary) != source.sha256:
                temporary.unlink(missing_ok=True)
                raise CustodyError(f"downloaded {source.file} does not match its pinned SHA-256")
            temporary.replace(target)
        if sha256_of(target) != source.sha256:
            raise CustodyError(f"{target} does not match its pinned SHA-256")
        paths[source.key] = target
    return paths


def manifest(sources: Sequence[Source], extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The custody block a receipt embeds."""
    return {"sources": {s.key: asdict(s) for s in sources}, **(extra or {})}
