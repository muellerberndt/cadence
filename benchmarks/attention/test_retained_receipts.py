"""The attention chambers' retained receipts, on archived bytes alone.

Every receipt under ``results/`` keeps its canonical form and digest, and every source it binds
(the library and the scripts as they ran) is an archived file named by its content hash under
``results/sources/``, or a retained result. These checks read no live brain: a receipt says what
was measured on the code it binds, and the library it is verified under may have moved on. Reading
one registered receipt's recorded arithmetic with its archived checker is
``python benchmarks/provenance.py --historical <receipt>``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import cadence as cd

RESULTS = Path(__file__).with_name("results")
ARCHIVE = RESULTS / "sources"
# bound by a summary and never retained: the full result it summarizes, over a size limit
NEVER_RETAINED = {"benchmarks/attention/results/continuous-relevance-capacity.json"}
RECEIPTS = sorted(RESULTS.glob("*.json"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stored(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


@pytest.mark.parametrize("path", RECEIPTS, ids=lambda path: path.name)
def test_every_source_a_receipt_binds_is_archived(path: Path) -> None:
    retained = {digest(result) for result in RESULTS.glob("*.json")}
    for row in stored(path).get("source", {}).get("files", []):
        if row["path"] in NEVER_RETAINED:
            continue
        archived = ARCHIVE / (row["sha256"] + ".txt")
        assert row["sha256"] in retained or digest(archived) == row["sha256"], row["path"]


@pytest.mark.parametrize(
    "path", [p for p in RECEIPTS if "digest" in stored(p)], ids=lambda path: path.name
)
def test_a_receipt_keeps_its_canonical_form_and_digest(path: Path) -> None:
    valid, message = cd.Receipt.verify(path)
    assert valid, message
