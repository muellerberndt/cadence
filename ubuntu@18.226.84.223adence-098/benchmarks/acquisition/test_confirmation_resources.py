"""Concurrent archive removal must not abort an unrelated founder."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def load_online(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    spec = importlib.util.spec_from_file_location(
        "online_resource_test", Path(__file__).with_name("online_curriculum.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_archive_disappearing_between_file_check_and_stat_is_ignored(monkeypatch):
    module = load_online(monkeypatch)

    class ArchivedSibling:
        def is_file(self):
            return True

        def stat(self):
            raise FileNotFoundError("completed sibling was archived")

    live = SimpleNamespace(is_file=lambda: True, stat=lambda: SimpleNamespace(st_size=12))
    root = SimpleNamespace(rglob=lambda pattern: iter([ArchivedSibling(), live]))
    assert module.tree_bytes(root) == 12


def test_storage_inspection_errors_are_not_silently_admitted(monkeypatch):
    module = load_online(monkeypatch)

    def denied():
        raise PermissionError("cannot inspect artifact storage")

    inaccessible = SimpleNamespace(is_file=lambda: True, stat=denied)
    root = SimpleNamespace(rglob=lambda pattern: iter([inaccessible]))
    with pytest.raises(PermissionError):
        module.tree_bytes(root)


@pytest.mark.parametrize("seeds", ((1, 2, 3, 4), (1, 2, 3, 4, 4), (0, 2, 3, 4, 5),
                                    (-1, 2, 3, 4, 5), (True, 2, 3, 4, 5)))
def test_confirmation_rejects_nonfresh_founders_before_creating_attempt(
    monkeypatch, tmp_path, seeds
):
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    spec = importlib.util.spec_from_file_location(
        "confirmation_admission_test", Path(__file__).with_name("online_half_step_confirmation.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "protocol.json").write_text(json.dumps({"brain_seed": 0}))
    attempt = tmp_path / "new-attempt"
    with pytest.raises(ValueError, match="five distinct fresh"):
        module.launch(attempt, reference, founder_seeds=seeds)
    assert not attempt.exists()


@pytest.mark.parametrize("mutation", ("failed", "unbound", "changed-task", "changed-cap"))
def test_confirmation_rejects_unqualified_reference_before_reading_panels(
    monkeypatch, tmp_path, mutation
):
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    spec = importlib.util.spec_from_file_location(
        "confirmation_reference_test", Path(__file__).with_name("online_half_step_confirmation.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reference = tmp_path / "reference"
    reference.mkdir()
    protocol_file = reference / "protocol.json"
    protocol = {"brain_seed": 0, "relations_protocol_sha256": "0" * 64}
    if mutation == "changed-cap":
        protocol.update(
            relations_protocol_sha256=module.relations.protocol_hash(),
            old_order=[0], mixed_order=[0] * 128, old_cap=2, mixed_cap=128,
        )
    protocol_file.write_text(json.dumps(protocol))
    (reference / "summary.json").write_text(json.dumps({
        "passed": mutation != "failed",
        "protocol_sha256": "0" * 64 if mutation == "unbound"
        else module.harness.sha256(protocol_file),
    }))

    def forbidden_panel(*args, **kwargs):
        raise AssertionError("unqualified reference reached panel access")

    monkeypatch.setattr(module.relations, "load_panel", forbidden_panel)
    attempt = tmp_path / "new-attempt"
    message = {
        "changed-task": "changed.*relation task",
        "changed-cap": "declared cap differs",
    }.get(mutation, "passed source-bound")
    with pytest.raises(ValueError, match=message):
        module.launch(attempt, reference)
    assert not attempt.exists()


def test_archiving_seed_one_preserves_seed_ten_and_its_active_stages(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    spec = importlib.util.spec_from_file_location(
        "confirmation_archive_test", Path(__file__).with_name("online_half_step_confirmation.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "custody").mkdir()
    for seed in (1, 10):
        for suffix in ("", "-old4", "-mixed"):
            folder = tmp_path / f"seed-{seed}{suffix}"
            folder.mkdir()
            (folder / "progress.json").write_text(str(seed))
    receipt = module.archive_founder(tmp_path, 1)
    assert set(receipt["original_member_sha256"]) == {
        f"seed-1{suffix}/progress.json" for suffix in ("", "-old4", "-mixed")
    }
    assert receipt["every_original_member_verified"] is True
    for suffix in ("", "-old4", "-mixed"):
        assert not (tmp_path / f"seed-1{suffix}").exists()
        assert (tmp_path / f"seed-10{suffix}/progress.json").read_text() == "10"
