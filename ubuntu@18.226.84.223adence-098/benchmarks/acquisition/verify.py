"""Replay saved phase admissions and independently audit original equation defects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from extract import sha256, write_json
from run import free_recall, independent_residual, sources

from cadence import Brain, BrainState


def verify(folder, *, allow_source_drift=False):
    protocol = json.loads((folder / "protocol.json").read_text())
    summary = json.loads((folder / "summary.json").read_text())
    if summary["protocol_sha256"] != sha256(folder / "protocol.json"):
        raise ValueError("summary protocol hash differs")
    if [item["recipe"] for item in summary["outcomes"]] != protocol["recipes"]:
        raise ValueError("scheduled recipe census differs")
    expected_cases = {
        f"{outcome['recipe']}-{stage['examples']}": stage
        for outcome in summary["outcomes"]
        for stage in outcome["stages"]
    }
    actual_cases = {p.parent.name for p in folder.glob("*-*/receipt.json")}
    if actual_cases != set(expected_cases):
        raise ValueError("summary/case completeness differs")
    if (folder / "source/run.py").exists():
        if sha256(folder / "source/run.py") != protocol["harness_sha256"]:
            raise ValueError("frozen harness hash differs")
        if sha256(folder / "source/extract.py") != protocol["extractor_sha256"]:
            raise ValueError("frozen extractor hash differs")
        for name, digest in protocol["library_sources"].items():
            if sha256(folder / "source/library/cadence" / name) != digest:
                raise ValueError("frozen library source hash differs")
    source_match = sources() == protocol["library_sources"]
    if not source_match and not allow_source_drift:
        raise ValueError("library source differs; run with frozen source/library on PYTHONPATH")
    fixture = folder / "source/fixture/school.npz"
    if not fixture.exists():
        fixture = Path(__file__).parent / "fixture/school.npz"
    if sha256(fixture) != protocol["fixture_sha256"]:
        raise ValueError("fixture hash differs")
    results = []
    with np.load(fixture, allow_pickle=False) as arrays:
        full_inputs, full_labels = arrays["school_inputs"], arrays["school_labels"]
    for receipt_path in sorted(folder.glob("*-*/receipt.json")):
        case = receipt_path.parent
        receipt = json.loads(receipt_path.read_text())
        stage = expected_cases[case.name]
        if sha256(case / "initial.npz") != receipt["initial_sha256"]:
            raise ValueError("initial checkpoint hash differs")
        if sha256(case / "final.npz") != receipt["final_sha256"]:
            raise ValueError("final checkpoint hash differs")
        brain = Brain.load(case / "initial.npz")
        if "effective_recipes" in protocol:
            recipe = case.name.rsplit("-", 1)[0]
            effective = protocol["effective_recipes"].get(recipe, {})
            if effective.get("learning") != brain.learner.config.to_dict():
                raise ValueError("effective recipe differs from initial learner configuration")
        if receipt["learner_config"] != brain.learner.config.to_dict():
            raise ValueError("receipt learner configuration differs from initial checkpoint")
        count = len(receipt["labels"])
        positions = {2: [0, 9], 4: [0, 9, 17, 22], 24: list(range(24))}[count]
        inputs, labels = full_inputs[positions], full_labels[positions]
        if labels.tolist() != receipt["labels"]:
            raise ValueError("case labels differ from frozen fixture")
        drive = brain.stimulus(inputs, memory=False)
        max_report_deviation, accepted, refused, phase_checks = 0.0, 0, 0, 0
        for record in receipt["updates"]:
            phase_path = case / f"phases-{record['update']:04d}.npz"
            if sha256(phase_path) != record["phase_sha256"]:
                raise ValueError("raw phase hash differs")
            phases = {}
            with np.load(phase_path, allow_pickle=False) as values:
                for name, reading in record["phases"].items():
                    state = BrainState(
                        values[name + "_v"],
                        values[name + "_activation"],
                        values[name + "_adaptation"],
                        reading["steps"],
                    )
                    phases[name] = state
                    residual, cache = independent_residual(
                        brain, drive, state, labels, {"free": 0, "nudged": 1, "opposite": -1}[name]
                    )
                    phase_checks += count
                    deviation = float(np.abs(residual - reading["full_residual"]).max())
                    max_report_deviation = max(max_report_deviation, deviation)
                    if deviation > 1e-10 or np.max(np.abs(cache - reading["cache_defect"])) > 1e-10:
                        raise ValueError("phase readback differs from independent replay")
                    if (
                        record["report"].get("accepted", 1)
                        and getattr(brain.learner.config, "qualified", False)
                        and (
                            residual.max() > brain.learner.config.tolerance
                            or cache.max() > brain.learner.config.tolerance
                        )
                    ):
                        raise ValueError("accepted qualified phase fails original equations")
            if record["report"].get("accepted", 1):
                expected = {"free", "nudged", "opposite"}  # the lesson's three phases
                if set(phases) != expected:
                    raise ValueError("accepted lesson phase census differs")
                brain.learner.update(phases["free"], phases["nudged"], phases.get("opposite"))
                accepted += 1
            else:
                refused += 1
        final = Brain.load(case / "final.npz")
        for name in ("efficacy", "bias", "log_gain"):
            if not np.array_equal(getattr(brain.brain, name), getattr(final.brain, name)):
                raise ValueError("replayed final parameters differ")
        for name in ("velocity", "velocity_bias", "second_moment", "second_moment_bias"):
            if not np.array_equal(getattr(brain.learner, name), getattr(final.learner, name)):
                raise ValueError("replayed optimizer state differs")
        for name in ("updates", "contrast_updates"):
            if getattr(brain.learner, name) != getattr(final.learner, name):
                raise ValueError("replayed learner counters differ")
        if (
            accepted != receipt["accepted_updates"]
            or accepted * count != receipt["accepted_row_exposures"]
        ):
            raise ValueError("accepted exposure accounting differs")
        if len(receipt["updates"]) != receipt["attempted_updates"]:
            raise ValueError("attempted admission accounting differs")
        expected_work = {
            "phase_row_sweeps": sum(
                count
                * sum(
                    int(r["report"].get(name + "_steps", reading["steps"]))
                    for name, reading in r["phases"].items()
                )
                for r in receipt["updates"]
            ),
            "recall_row_sweeps": sum(r["work"]["row_sweeps"] for r in receipt["recall"]),
            "independent_phase_residual_checks": sum(
                len(r["phases"]) * count for r in receipt["updates"]
            ),
            "reported_phase_residual_checks": sum(
                r["report"].get("total_residual_checks", 0) for r in receipt["updates"]
            ),
            "reported_phase_row_residual_checks": sum(
                count * r["report"].get("total_residual_checks", 0) for r in receipt["updates"]
            ),
            "refused_recall_rows": sum(r["refusals"] for r in receipt["recall"]),
        }
        for name, value in expected_work.items():
            if name not in receipt["work"]:
                raise ValueError("mandatory work field missing: " + name)
            if receipt["work"][name] != value:
                raise ValueError("work accounting differs: " + name)
        recall = free_recall(final, inputs, labels)
        if recall["predictions"] != receipt["recall"][-1]["predictions"]:
            raise ValueError("final free predictions differ")
        for name in ("correct", "refusals", "status", "accepted_row_exposures"):
            value = receipt[name] if name in ("status", "accepted_row_exposures") else recall[name]
            if stage[name] != value:
                raise ValueError("summary outcome differs from replay: " + name)
        if stage["passed"] and (
            recall["refusals"] or recall["correct"] < (count if count < 24 else 18) or not accepted
        ):
            raise ValueError("summary promotes a failed acquisition gate")
        results.append(
            {
                "case": case.name,
                "accepted": accepted,
                "refused": refused,
                "max_phase_residual_deviation": max_report_deviation,
                "independent_phase_row_checks": phase_checks,
                "verification_query_work": recall["work"],
                "correct": recall["correct"],
                "refusals": recall["refusals"],
            }
        )
    result = {
        "schema": "cadence-acquisition-verification-v1",
        "passed": bool(results) and source_match and summary.get("library_sources_unchanged", True),
        "source_match": source_match,
        "cases": results,
        "verifier_sha256": sha256(__file__),
        "source_bound": source_match and summary.get("library_sources_unchanged", True),
        "harness_checkout_unchanged": summary.get("harness_source_unchanged"),
        "producing_harness_bytes_verified": (folder / "source/run.py").exists(),
        "unverified": [
            "phase trajectory generation",
            "microscope period-two trajectory",
            "earlier recall prefixes",
            "independent TRAIN/development panels",
            "saved continuation",
        ],
        "boundary": (
            "Equation/readback/update/counter/census replay audit. Explicit unverified checks "
            "remain separate; no proof of generalization. "
            "Source drift permits diagnostic replay only."
        ),
    }
    write_json(folder / "verification.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--allow-source-drift", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.folder, allow_source_drift=args.allow_source_drift)))


if __name__ == "__main__":
    main()
