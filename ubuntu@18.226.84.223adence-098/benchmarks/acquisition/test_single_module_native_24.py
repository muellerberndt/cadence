"""Native24 extension preserves its gate and seals unused movie panels."""

from __future__ import annotations

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent


@pytest.fixture
def extension(monkeypatch):
    for name in (
        "extract",
        "run",
        "relations",
        "relation_development",
        "continual",
        "online_curriculum",
        "native_online",
        "single_module_native",
        "single_module_native_24",
    ):
        spec = importlib.util.spec_from_file_location(name, ROOT / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
    return sys.modules["single_module_native_24"]


@pytest.mark.parametrize(
    "change",
    [
        {"seed": 1},
        {"lesson_caps": {"24": 8192}},
        {"selected_correct": {"24": 14}},
        {"independent_correct": {"independent_train": 3, "development": 1}},
        {"output_cap_mib": 320},
    ],
)
def test_native_protocol_does_not_weaken_gates_or_search(extension, change):
    value = deepcopy(extension.FIXED)
    value["orders_including_next_lesson"] = {"24": [0] * 4097}
    value.update(change)
    with pytest.raises(ValueError, match="source recipe/gate/bound"):
        extension.check_protocol(value)


def fixture(extension, tmp_path):
    root = tmp_path
    folder = root / "source/fixture"
    folder.mkdir(parents=True)
    data, panels = {}, {}
    for panel, count in (("school", 24), ("independent_train", 18), ("development", 19)):
        x = np.zeros((count, 650), dtype=np.float64)
        x[:, 0] = np.arange(count)
        y = np.arange(count, dtype=np.int64)
        data[panel + "_inputs"], data[panel + "_labels"] = x, y
        panels[panel] = [
            {"input_sha256": extension.native.harness.row_hash(row), "label": int(label)}
            for row, label in zip(x, y, strict=True)
        ]
    data["heldout_inputs"] = np.array([np.nan])
    data["heldout_labels"] = np.array([np.nan])
    np.savez(folder / "school.npz", **data)
    (folder / "provenance.json").write_text(json.dumps({"panels": panels}))
    return root, folder, data


def test_school_loader_keeps_independent_and_heldout_arrays_lazy(extension, tmp_path):
    root, folder, _ = fixture(extension, tmp_path)
    panels = extension.load_fixture(folder)
    assert panels.root == root
    assert set(panels) == {"school_inputs", "school_labels"}
    assert not panels.decoded
    with pytest.raises(KeyError, match="never decodes"):
        panels["heldout_inputs"]
    assert not (root / "independent-read-census.json").exists()


def test_independent_label_read_requires_actual_selected_screen(extension, tmp_path):
    root, folder, _ = fixture(extension, tmp_path)
    panels = extension.load_fixture(folder)
    (root / "selected-24").mkdir()
    (root / "selected-24/state.json").write_text(
        json.dumps(
            {
                "selected_passed": False,
                "recall": [{"correct": 0, "refusals": 0}, {"correct": 17, "refusals": 0}],
                "work": {"refused_teacher_calls": 0, "unqualified_teacher_calls": 0},
            }
        )
    )
    with pytest.raises(ValueError, match="before native24 screen"):
        panels["development_labels"]
    assert set(panels) == {"school_inputs", "school_labels"}
    assert not panels.decoded


def test_independent_panel_decodes_once_after_screen_without_heldout(extension, tmp_path):
    root, folder, data = fixture(extension, tmp_path)
    panels = extension.load_fixture(folder)
    (root / "selected-24").mkdir()
    (root / "selected-24/final.npz").write_bytes(b"bound final checkpoint fixture")
    (root / "selected-24/state.json").write_text(
        json.dumps(
            {
                "selected_passed": True,
                "completed_lessons": 192,
                "recall": [{"correct": 0, "refusals": 0}, {"correct": 18, "refusals": 0}],
                "work": {"refused_teacher_calls": 0, "unqualified_teacher_calls": 0},
            }
        )
    )
    assert np.array_equal(panels["development_inputs"], data["development_inputs"])
    assert np.array_equal(panels["development_labels"], data["development_labels"])
    assert panels.decoded["development"]["decodes"] == 1
    assert not panels.decoded["development"]["labels_used_for_teaching"]
    assert set(panels.decoded) == {"development"}
    assert not json.loads((root / "independent-read-census.json").read_text())["heldout_decoded"]
