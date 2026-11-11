"""Portable display identities and honest scope, with no Cadence imports or solves."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).parent


def read(name: str) -> dict:
    return json.loads((HERE / name).read_text())


def test_bundled_file_identities_and_no_host_paths() -> None:
    manifest = read("bundle-manifest.json")
    assert manifest["recorded_library_version"] == "0.73.0"
    for artifact in manifest["artifacts"]:
        path = HERE / artifact["path"]
        assert path.is_file()
        data = path.read_bytes()
        assert hashlib.sha256(data).hexdigest() == artifact["sha256"]
        assert len(data) == artifact["bytes"]
        if path.suffix in {".json", ".html", ".md"}:
            assert b"/Users/" not in data
            assert b"/tmp/" not in data


def test_complete_census_failures_and_historical_scope() -> None:
    manifest = read("bundle-manifest.json")
    expected = {"half-rate-data.json": 1, "quarter-rate-data.json": 3,
                "demo-data.json": 4, "normalized-data.json": 5}
    count = 0
    for name, passed in expected.items():
        data = read(name)
        assert data["schema"] == "cadence.acquisition-demo.v1"
        assert data["source"]["version"] == "0.73.0"
        assert len(data["founders"]) == data["founder_denominator"] == 5
        assert sum(founder["passed"] is True for founder in data["founders"]) == passed
        assert data["bundle_provenance"]["raw_phases_checkpoints_corpus_bundled"] is False
        assert all("stages" in founder and "work" in founder for founder in data["founders"])
        count += len(data["founders"])
    assert count == 20
    assert len(manifest["campaigns"]) == 4
    quarter = next(f for f in read("quarter-rate-data.json")["founders"] if f["seed"] == 6)
    assert quarter["passed"] is False
    assert quarter["final"]["new_correct"] is None
    failed = next(f for f in read("demo-data.json")["founders"] if f["seed"] == 19)
    assert failed["passed"] is False and failed["final"]["old_correct"] == 2
    assert failed["final"]["heldout_correct"] is None
    ledger = read("research-ledger.json")
    assert ledger["issue_closed"] is False
    native = ledger["native_selected_acquisition_extension"]
    assert native["independent_train"]["correct"] == 6
    assert native["development"]["correct"] == 0
    assert ledger["actual_component_coverage_continuation"]["whole_issue_passed"] is False
    assert ledger["portable_bundle"]["raw_artifacts_bundled"] is False


def test_external_identity_index_has_no_claim_of_local_raw_archive() -> None:
    index = read("receipt-index.json")
    assert index["artifacts"]
    assert all(item["availability"] == "external_not_bundled" for item in index["artifacts"])
    assert all("path" not in item for item in index["artifacts"])
    assert all(re.fullmatch(r"[a-f0-9]{64}", item["sha256"]) for item in index["artifacts"])
    html = (HERE / "index.html").read_text()
    match = re.search(
        r'<script id="receipt-index-data" type="application/json">(.*?)</script>',
        html, re.DOTALL,
    )
    assert match is not None and json.loads(match.group(1)) == index
    for campaign in read("bundle-manifest.json")["campaigns"]:
        data = read(campaign["file"])
        original = data["bundle_provenance"]["original_export"]
        assert any(item["sha256"] == original["sha256"] for item in index["artifacts"])
