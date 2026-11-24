"""A checkout holds every tracked file's committed bytes, whatever core.autocrlf says: receipts
and ledgers hash sources and results byte for byte (.gitattributes)."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = "benchmarks/attention/results/sources/"
# what git reports for a file whose line endings no setting of the machine converts
KEPT = {"-text", "text eol=lf", "text=auto eol=lf"}


def tracked() -> dict[str, tuple[str, str, str]]:
    """Each tracked file's line endings as committed and as checked out, and its line-ending
    attributes, as ``git ls-files --eol`` reads them."""
    if sys.platform == "emscripten" or shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("line-ending attributes are read from a git checkout")
    listing = subprocess.run(["git", "ls-files", "--eol"], cwd=ROOT, capture_output=True,
                             text=True, check=True)
    out = {}
    for line in listing.stdout.splitlines():
        told, path = line.split("\t", 1)
        committed, checked_out, attributes = told.split(None, 2)
        out[path] = (committed[2:], checked_out[2:], attributes.strip()[5:])
    return out


def test_no_tracked_file_takes_its_line_endings_from_the_machine():
    files = tracked()
    open_to_it = sorted(path for path, (_, _, attributes) in files.items()
                        if attributes not in KEPT)
    assert not open_to_it, f"{len(open_to_it)} files, among them {open_to_it[:5]}"
    converted = sorted(path for path, (committed, checked_out, _) in files.items()
                       if committed != checked_out)
    assert not converted, f"{len(converted)} files, among them {converted[:5]}"


def test_the_archived_sources_are_never_converted():
    archived = {path: attributes for path, (_, _, attributes) in tracked().items()
                if path.startswith(ARCHIVE)}
    assert archived and set(archived.values()) == {"-text"}
