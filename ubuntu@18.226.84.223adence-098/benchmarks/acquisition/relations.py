"""Frozen, identifiable cue-to-action relations for acquisition and retention.

This is a controlled association microscope, not a game-policy dataset. A cue
identity determines its target, and independent nuisance never changes it. The
centroid/nearest-example controls below establish that the task is identifiable;
their feed-forward answers are not Cadence results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

TASK_SEED = 11020261003
INPUTS = 650
ACTIONS = 36
FAMILIES = 24
CUE_WIDTH = 16
CUE_AMPLITUDE = 1.0
NUISANCE_START = FAMILIES * CUE_WIDTH
NUISANCE_WIDTH = 64
NUISANCE_AMPLITUDE = 0.1
PANEL_COUNTS = {"train": 4, "development": 2, "heldout": 2}
SPLIT_KEYS = {"train": 1, "development": 2, "heldout": 3}
TWO_FAMILIES = (0, 9)
OLD_FAMILIES = (0, 9, 17, 22)
NEW_FAMILIES = tuple(i for i in range(FAMILIES) if i not in OLD_FAMILIES)
DEVELOPMENT_FOUNDER = 0
CONFIRMATION_FOUNDERS = (1, 2, 3, 4, 5)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def labels_by_family():
    """A fixed shuffled injection, never a copied sensory-coordinate readout."""
    rng = np.random.default_rng(np.random.SeedSequence([TASK_SEED, 0]))
    return rng.permutation(ACTIONS)[:FAMILIES].astype("<i8")


def protocol():
    """Return the complete task contract, including its generating source pin."""
    return {
        "schema": "cadence-controlled-relations-v1",
        "generator_sha256": sha256(__file__),
        "task_seed": TASK_SEED,
        "inputs": INPUTS,
        "actions": ACTIONS,
        "family_count": FAMILIES,
        "family_to_label": labels_by_family().tolist(),
        "cue": {
            "rule": "family i has coordinates [16*i,16*(i+1)) at +1; others zero",
            "width": CUE_WIDTH,
            "amplitude": CUE_AMPLITUDE,
        },
        "nuisance": {
            "rule": "independent uniform draws; irrelevant to the deterministic target",
            "coordinates": [NUISANCE_START, NUISANCE_START + NUISANCE_WIDTH],
            "amplitude": NUISANCE_AMPLITUDE,
            "rng": "numpy.default_rng(SeedSequence([task_seed, split_key]))",
            "split_keys": SPLIT_KEYS,
        },
        "panel_observations_per_family": PANEL_COUNTS,
        "panel_order": "ascending family, then independent nuisance instance",
        "two_families": list(TWO_FAMILIES),
        "four_and_old_families": list(OLD_FAMILIES),
        "new_families": list(NEW_FAMILIES),
        "development_founder": DEVELOPMENT_FOUNDER,
        "confirmation_founders": list(CONFIRMATION_FOUNDERS),
        "answer": {
            "entry": "Brain.compose(650,36,modules=(32,16),observers=())",
            "teacher_absent": True,
            "whole_state_residual_tolerance": 0.003,
            "zero_refusals": True,
            "qualification": "original complete neural equations; no answer clamps",
        },
        "gates": {
            "two_and_four": "perfect TRAIN free recall with accepted teaching and founder gain",
            "24_screen": (
                "at least 18/24 families; all four TRAIN instances correct per credited family"
            ),
            "independent_development": "at least 36/48 correct; balanced macro accuracy >= .75",
            "heldout_confirmation": "all five fresh founders >= 36/48; no candidate changes",
            "retention": "old >= 3/4 and new >= 15/20 families on independent cue-present probes",
            "family_credit": "all observations in the relevant balanced panel correct",
        },
        "experience": {
            "repeated_exposures": "allowed and counted as presentations, not new relations",
            "teacher": "observed current cue-to-label relation, no synthetic reward",
            "development_use": "declared recipe selection; preserve every failed attempt",
            "heldout_use": "unread until matching frozen recipe passes development",
            "founders": "confirmation models are fresh; task mapping and panels remain fixed",
        },
        "retention_boundary": {
            "cue_present": True,
            "reset": "clear live state/Trace and hippocampal fast F; retain slow parameters/C",
            "sequential": "acquire old four; teach disjoint new20; query both with teacher absent",
            "rehearsal": "separate declared arm; charge all old/new replay and query work",
            "checkpoint": "save-load predictions and actual next accepted/refused learning outcome",
        },
        "boundaries": [
            "controlled learned associations; no NES competence or natural category claim",
            "not a vanished-cue working-memory capacity test",
            "SynapticMemory normalized-delta writes do not certify local contrast learning",
            "competent centroid/nearest-example controls are baselines, never brain results",
            "historical movie-frame acquisition/retention failure remains separate evidence",
        ],
    }


def protocol_hash():
    return hashlib.sha256(_json_bytes(protocol())).hexdigest()


def _check_heldout_unlock(receipt):
    if receipt is None:
        raise ValueError("heldout sealed: matching frozen-recipe development receipt required")
    if isinstance(receipt, (str, Path)):
        receipt = json.loads(Path(receipt).read_text())
    if (
        receipt.get("relations_protocol_sha256") != protocol_hash()
        or receipt.get("development_passed") is not True
        or receipt.get("zero_refusals") is not True
        or receipt.get("recipe_frozen_before_development") is not True
        or not isinstance(receipt.get("recipe_sha256"), str)
        or len(receipt["recipe_sha256"]) != 64
    ):
        raise ValueError("heldout unlock does not bind a passed development and frozen recipe")


def load_panel(name, families=None, *, development_receipt=None):
    """Return read-only inputs/labels/families/instances for a declared split.

    TRAIN has four observations per family. Development/test have two fresh
    nuisance draws per family. No heldout values are generated for other names.
    A family filter preserves ascending family order.
    """
    if name not in PANEL_COUNTS:
        raise ValueError(f"unknown panel {name!r}")
    if name == "heldout":
        _check_heldout_unlock(development_receipt)
    count = PANEL_COUNTS[name]
    family_ids = np.repeat(np.arange(FAMILIES, dtype="<i8"), count)
    instances = np.tile(np.arange(count, dtype="<i8"), FAMILIES)
    inputs = np.zeros((len(family_ids), INPUTS), dtype="<f8")
    for family in range(FAMILIES):
        inputs[family_ids == family, family * CUE_WIDTH : (family + 1) * CUE_WIDTH] = CUE_AMPLITUDE
    rng = np.random.default_rng(np.random.SeedSequence([TASK_SEED, SPLIT_KEYS[name]]))
    inputs[:, NUISANCE_START : NUISANCE_START + NUISANCE_WIDTH] = rng.uniform(
        -NUISANCE_AMPLITUDE, NUISANCE_AMPLITUDE, (len(inputs), NUISANCE_WIDTH)
    )
    labels = labels_by_family()[family_ids]
    if families is not None:
        requested = tuple(int(i) for i in families)
        if len(set(requested)) != len(requested) or any(i < 0 or i >= FAMILIES for i in requested):
            raise ValueError("families must be distinct IDs in [0,24)")
        mask = np.isin(family_ids, requested)
        inputs, labels, family_ids, instances = (
            a[mask] for a in (inputs, labels, family_ids, instances)
        )
    result = {"inputs": inputs, "labels": labels, "families": family_ids, "instances": instances}
    for array in result.values():
        array.setflags(write=False)
    return result


def competent_controls():
    """Independent TRAIN-fit controls, with no development fitting or test access."""
    train, development = load_panel("train"), load_panel("development")
    centers = np.stack([train["inputs"][train["families"] == i].mean(0) for i in range(FAMILIES)])
    family_labels = labels_by_family()
    result = {}
    for name, panel in (("train", train), ("development", development)):
        distances = ((panel["inputs"][:, None, :] - train["inputs"][None, :, :]) ** 2).sum(2)
        centroid_distances = ((panel["inputs"][:, None, :] - centers[None, :, :]) ** 2).sum(2)
        nearest = train["labels"][distances.argmin(1)]
        centroid = family_labels[centroid_distances.argmin(1)]
        result[name] = {
            "rows": len(panel["labels"]),
            "nearest_example_correct": int((nearest == panel["labels"]).sum()),
            "centroid_correct": int((centroid == panel["labels"]).sum()),
            "target_law_correct": int(
                (
                    family_labels[
                        panel["inputs"][:, :NUISANCE_START]
                        .reshape(-1, FAMILIES, CUE_WIDTH)
                        .sum(2)
                        .argmax(1)
                    ]
                    == panel["labels"]
                ).sum()
            ),
        }
    result["geometry"] = {
        "different_cue_distance": float(np.sqrt(2 * CUE_WIDTH) * CUE_AMPLITUDE),
        "maximum_same_cue_distance": float(2 * NUISANCE_AMPLITUDE * np.sqrt(NUISANCE_WIDTH)),
        "legal_action_uniform_chance": 1 / ACTIONS,
        "observed_label_uniform_chance": 1 / FAMILIES,
    }
    result["boundary"] = "feed-forward identifiability controls only; heldout ungenerated/unread"
    return result


def freeze(out):
    """Write <100KB TRAIN/dev fixtures, protocol and generating source; no test values."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    task = protocol()
    (out / "protocol.json").write_bytes(_json_bytes(task))
    shutil.copy2(__file__, out / "relations.py")
    provenance = {
        "schema": "cadence-controlled-relations-fixture-v1",
        "relations_protocol_sha256": protocol_hash(),
        "generator_sha256": sha256(__file__),
        "panels": {},
        "heldout": "sealed; no values generated or written",
    }
    for name in ("train", "development"):
        panel = load_panel(name)
        np.savez_compressed(out / f"{name}.npz", **panel)
        provenance["panels"][name] = {
            "rows": len(panel["labels"]),
            "instances_per_family": PANEL_COUNTS[name],
            "fixture_sha256": sha256(out / f"{name}.npz"),
            "array_sha256": {key: array_hash(value) for key, value in panel.items()},
        }
    provenance["competent_controls"] = competent_controls()
    (out / "provenance.json").write_bytes(_json_bytes(provenance))
    data_bytes = sum((out / f"{name}.npz").stat().st_size for name in ("train", "development"))
    if data_bytes >= 100_000:
        raise ValueError(f"fixture data exceeded the declared 100KB boundary: {data_bytes}")
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    provenance = freeze(args.out)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "protocol_sha256": protocol_hash(),
                "controls": provenance["competent_controls"],
            }
        )
    )


if __name__ == "__main__":
    main()
