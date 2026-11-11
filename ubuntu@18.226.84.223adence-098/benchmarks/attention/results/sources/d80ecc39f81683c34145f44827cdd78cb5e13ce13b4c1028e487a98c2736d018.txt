"""Verify retained admission decisions, source bytes and receipt digests.

Historical source versions are content-addressed in results/sources. This
checks the actual old bytes rather than pretending the runner never changed.
It verifies custody and arithmetic; it does not rerun behavioral acquisition.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path, PurePosixPath
from typing import Any

from cadence.receipts import Receipt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SPEC = importlib.util.spec_from_file_location("attention_admission", HERE / "admission.py")
assert SPEC is not None and SPEC.loader is not None
admission = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = admission
SPEC.loader.exec_module(admission)


def check(body: dict[str, Any]) -> str | None:
    protocol = body["protocol"]
    founders = body["founders"]
    if [f["seed"] for f in founders] != protocol["development_seeds"]:
        return "development founders differ"
    for founder in founders:
        training, evaluation = founder["training"], founder["evaluation"]
        if training["completed"] != (training["rounds"] == protocol["rounds"]):
            return "training completion differs"
        for branch in evaluation["branches"].values():
            if admission.relevance.scores(branch["answers"]) != branch["scores"]:
                return "behavioral scores differ"
        passed = training["completed"] and admission.relevance.acquisition_gate(
            evaluation, protocol["acquisition_floor"])
        if passed != founder["passed"]:
            return "founder gate differs"
    passed = all(f["passed"] for f in founders)
    if body["passed"] != passed:
        return "aggregate gate differs"
    expected = ("admitted_for_development_relevance_tests" if passed else
                "stop_behavioral_candidates; acquisition_failure_not_localized_to_retrieval")
    if body["decision"] != expected:
        return "admission decision differs"
    if body["candidate_rounds_used"] != 0 or body["confirmation_seeds_used"]:
        return "an admission receipt must not report candidate or confirmation work"
    return None


def verify(path: Path) -> tuple[bool, str]:
    receipt = Receipt.read(path)
    sources = []
    for entry in receipt.source["files"]:
        relative = PurePosixPath(entry["path"])
        digest = entry["sha256"]
        if (relative.is_absolute() or ".." in relative.parts or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)):
            return False, "invalid source manifest path or digest"
        snapshot = HERE / "results" / "sources" / (digest + ".txt")
        source = snapshot if snapshot.exists() else ROOT / relative
        sources.append((str(relative), source))
    arithmetic = check
    modules = {"cadence-attention-acquisition-repair/1": "repair_admission.py",
               "cadence-attention-content-control/1": "content_control.py",
               "cadence-attention-heads-admission/1": "heads_admission.py",
               "cadence-attention-heads-relevance/1": "heads_relevance.py",
               "cadence-attention-heads-capacity-diagnostic/1": "diagnostic_verification.py"}
    if receipt.kind in modules:
        spec = importlib.util.spec_from_file_location(
            "attention_receipt_verifier", HERE / modules[receipt.kind])
        assert spec is not None and spec.loader is not None
        repair = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = repair
        spec.loader.exec_module(repair)
        if receipt.kind in ("cadence-attention-heads-admission/1",
                            "cadence-attention-heads-relevance/1"):
            protocol_file = receipt.body.get("protocol_file", "heads-protocol.json")
            if protocol_file not in ("heads-protocol.json", "heads-longer-protocol.json",
                                     "heads-fixed-read-protocol.json"):
                return False, "unknown heads protocol"
            owner = repair if receipt.kind.endswith("admission/1") else repair.admission
            owner.PROTOCOL = HERE / protocol_file
        arithmetic = repair.check
    return Receipt.verify(path, sources=sources, check=arithmetic)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    ok, message = verify(args.receipt)
    print(message)
    raise SystemExit(0 if ok else 1)
