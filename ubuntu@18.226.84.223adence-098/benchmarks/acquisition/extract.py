"""Extract the small native witness school without copying its parent corpus."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def row_hash(value):
    return hashlib.sha256(np.asarray(value, dtype="<f8").tobytes()).hexdigest()


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def extract(witness, verification, selection, out):
    """Freeze selected rows, disjoint checks, and a sealed balanced test panel."""
    payload = json.loads(witness.read_text())
    native = json.loads(verification.read_text())
    selected = json.loads(selection.read_text())
    if (
        payload["source"]["inputs"] != 650
        or payload["source"]["window"] != 12
        or native.get("synced") is not True
        or native.get("deaths") != 0
    ):
        raise ValueError("native sensor/verification boundary differs")
    if selected["dataset_sha256"] != sha256(witness) or selected["verification_sha256"] != sha256(
        verification
    ):
        raise ValueError("parent fixture hashes differ from the independent receipt")
    rows = payload["bootstrap"] + payload["development"]["examples"]
    rows.sort(key=lambda row: row["frame"])
    frames = np.array([row["frame"] for row in rows], dtype=np.int64)
    if np.any(np.diff(frames) <= 0) or frames[-1] + 12 > native["frames"]:
        raise ValueError("frames are duplicate or outside native receipt")
    actions = tuple(itertools.product((-1, 0, 1), (-1, 0, 1), (0, 1), (0, 1)))
    inputs = np.asarray([row["inputs"] for row in rows], dtype="<f8")
    labels = np.array([actions.index(tuple(row["action"])) for row in rows], dtype=np.int64)
    if inputs.shape != (len(rows), 650) or not np.isfinite(inputs).all():
        raise ValueError("invalid sensory values")
    chosen = [item["train_row"] for item in selected["selection_rows"]]
    train_end, dev_end = int(len(rows) * 0.6), int(len(rows) * 0.8)
    if len(chosen) != 24 or len(set(chosen)) != 24 or len(set(labels[chosen])) != 24:
        raise ValueError("24 distinct selected TRAIN relations required")
    for item, index in zip(selected["selection_rows"], chosen, strict=True):
        if not 0 <= index < train_end or (
            int(frames[index]),
            int(labels[index]),
            row_hash(inputs[index]),
        ) != (item["frame"], item["label"], item["input_sha256"]):
            raise ValueError("selected row/frame/label/input identity differs")
    families = sorted(set(labels[chosen].tolist()))

    def first_per_family(start, end, excluded=()):
        return [
            next(
                index
                for index in range(start, end)
                if labels[index] == label and index not in excluded
            )
            for label in families
            if any(labels[index] == label and index not in excluded for index in range(start, end))
        ]

    independent = first_per_family(0, train_end, set(chosen))
    development = first_per_family(train_end, dev_end)
    test = first_per_family(dev_end, len(rows))
    common = sorted(set(labels[independent]) & set(labels[development]) & set(labels[test]))
    panels = {
        "school": chosen,
        "independent_train": independent,
        "development": development,
        "confirmation_train": [i for i in independent if labels[i] in common],
        "confirmation_development": [i for i in development if labels[i] in common],
        "heldout": [i for i in test if labels[i] in common],
    }
    arrays, identities = {}, {}
    for name, indexes in panels.items():
        arrays[name + "_inputs"] = inputs[indexes]
        arrays[name + "_labels"] = labels[indexes]
        identities[name] = [
            {
                "parent_row": i,
                "frame": int(frames[i]),
                "label": int(labels[i]),
                "input_sha256": row_hash(inputs[i]),
            }
            for i in indexes
        ]
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / "school.npz", **arrays)
    provenance = {
        "schema": "cadence-native-school-v1",
        "dtype": "little-endian float64",
        "parent_witness_sha256": sha256(witness),
        "parent_verification_sha256": sha256(verification),
        "selection_receipt_sha256": sha256(selection),
        "extractor_sha256": sha256(__file__),
        "parent_source": payload["source"],
        "native_verification": native,
        "split": {
            "rule": "chronological 60/20/20",
            "rows": len(rows),
            "train_end": train_end,
            "development_end": dev_end,
        },
        "panels": identities,
        "two_positions": [0, 9],
        "four_positions": [0, 9, 17, 22],
        "confirmation_families": [int(i) for i in common],
        "independent_train_missing_families": sorted(set(families) - set(labels[independent])),
        "fixture_sha256": sha256(out / "school.npz"),
        "selection_rule": (
            "historical exact school; checks use first other chronological row per action"
        ),
        "heldout_rule": "sealed TEST panel; never used by development runner",
        "boundary": (
            "One recorded no-damage FCEUX movie; no fresh engine harvest, gameplay gain "
            "or source transfer claim."
        ),
    }
    # NumPy integers in the missing-family set need ordinary JSON values.
    provenance["independent_train_missing_families"] = [
        int(i) for i in provenance["independent_train_missing_families"]
    ]
    write_json(out / "provenance.json", provenance)
    return {name: len(indexes) for name, indexes in panels.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    workspace = Path(__file__).resolve().parents[3]
    parser.add_argument(
        "--witness", type=Path, default=workspace / "cadence-castlevania/data/witnesses_6630M.json"
    )
    parser.add_argument(
        "--verification",
        type=Path,
        default=workspace / "cadence-castlevania/runs/harvest_6630M/verify.json",
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=workspace
        / "plan/evidence/cadence-nes070-20261002/independent-acquisition-summary.json",
    )
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "fixture")
    args = parser.parse_args()
    print(json.dumps(extract(args.witness, args.verification, args.selection, args.out)))


if __name__ == "__main__":
    main()
