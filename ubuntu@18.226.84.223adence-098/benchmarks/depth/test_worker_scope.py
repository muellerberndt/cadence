"""Reused process workers must run the declared depth and restore their constructor."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

import cadence as cd

spec = importlib.util.spec_from_file_location("depth_worker", Path(__file__).with_name("memory.py"))
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)


def test_reused_worker_does_not_multiply_depth_between_jobs(monkeypatch):
    original = cd.Brain.compose.__func__
    depths = []

    def founder(seed):
        brain = cd.Brain.compose(2, 2, backend="cpu", modules=(4,), seed=seed, episodic=False, recall=False)
        populations = brain.connectome.populations
        depths.append(1 + sum(name.startswith("module_") and "/" not in name
                              for name in populations))
        return {"passed": False, "gates": {"synthetic_control": False}}

    monkeypatch.setattr(subject, "benchmark", lambda name: SimpleNamespace(run_founder=founder))
    for depth in (2, 2, 3, 2):
        subject.run_cell("look_back", depth, 0, None)
        assert cd.Brain.compose.__func__ is original
    assert depths == [2, 2, 3, 2]


def test_refused_worker_restores_the_original_constructor(monkeypatch):
    original = cd.Brain.compose.__func__

    def refused(seed):
        raise RuntimeError("declared numerical refusal")

    monkeypatch.setattr(subject, "benchmark", lambda name: SimpleNamespace(run_founder=refused))
    with pytest.raises(RuntimeError, match="numerical refusal"):
        subject.run_cell("look_back", 3, 0, None)
    assert cd.Brain.compose.__func__ is original
