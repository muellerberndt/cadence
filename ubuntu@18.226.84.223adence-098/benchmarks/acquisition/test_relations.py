"""Guard family identifiability, split independence and sealed confirmation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "frozen_relations", Path(__file__).with_name("relations.py")
)
relations = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(relations)


def test_balanced_family_rule_and_independent_non_singleton_observations():
    train, development = relations.load_panel("train"), relations.load_panel("development")
    mapping = relations.labels_by_family()
    assert len(set(mapping)) == 24
    assert not np.array_equal(mapping, np.arange(24))
    hashes = set()
    for panel, count in ((train, 4), (development, 2)):
        assert panel["inputs"].shape == (24 * count, 650)
        assert np.array_equal(np.bincount(panel["families"]), np.full(24, count))
        assert np.array_equal(panel["labels"], mapping[panel["families"]])
        code = panel["inputs"][:, :384].reshape(-1, 24, 16)
        assert np.array_equal(code.sum(2).argmax(1), panel["families"])
        assert np.all(code.sum((1, 2)) == 16)
        assert np.all(panel["inputs"][:, 448:] == 0)
        assert np.all(np.abs(panel["inputs"][:, 384:448]) <= 0.1)
        assert not panel["inputs"].flags.writeable
        for observation in panel["inputs"]:
            digest = relations.array_hash(observation)
            assert digest not in hashes
            hashes.add(digest)


def test_competent_train_fit_controls_and_class_separation():
    report = relations.competent_controls()
    for name, rows in (("train", 96), ("development", 48)):
        assert report[name]["rows"] == rows
        for key in ("nearest_example_correct", "centroid_correct", "target_law_correct"):
            assert report[name][key] == rows
    assert (
        report["geometry"]["different_cue_distance"]
        > (report["geometry"]["maximum_same_cue_distance"])
    )


def test_panel_filter_preserves_balanced_identity_and_rejects_invalid_ids():
    panel = relations.load_panel("development", families=relations.OLD_FAMILIES)
    assert panel["inputs"].shape == (8, 650)
    assert tuple(np.unique(panel["families"])) == relations.OLD_FAMILIES
    assert np.array_equal(panel["instances"], np.tile((0, 1), 4))
    for ids in ((0, 0), (-1,), (24,)):
        with pytest.raises(ValueError, match="distinct IDs"):
            relations.load_panel("train", families=ids)


@pytest.mark.parametrize("mutated", (None, {}, {"development_passed": True}))
def test_heldout_refuses_before_input_generation(monkeypatch, mutated):
    original_zeros = relations.np.zeros

    def forbidden_inputs(*args, **kwargs):
        if args and args[0] == (48, 650):
            raise AssertionError("sealed panel inputs were generated")
        return original_zeros(*args, **kwargs)

    monkeypatch.setattr(relations.np, "zeros", forbidden_inputs)
    with pytest.raises(ValueError, match="heldout"):
        relations.load_panel("heldout", development_receipt=mutated)


def test_freeze_contains_no_heldout_arrays_and_binds_generating_source(tmp_path):
    target = tmp_path / "fixture"
    report = relations.freeze(target)
    assert not (target / "heldout.npz").exists()
    assert sum(p.stat().st_size for p in target.glob("*.npz")) < 100_000
    assert report["relations_protocol_sha256"] == relations.protocol_hash()
    assert relations.sha256(target / "relations.py") == relations.sha256(relations.__file__)
    frozen_protocol = json.loads((target / "protocol.json").read_text())
    assert frozen_protocol == relations.protocol()
    for name in ("train", "development"):
        expected = relations.load_panel(name)
        with np.load(target / f"{name}.npz") as archive:
            for key, array in expected.items():
                assert np.array_equal(array, archive[key])
                assert report["panels"][name]["array_sha256"][key] == relations.array_hash(array)
    with pytest.raises(FileExistsError):
        relations.freeze(target)
