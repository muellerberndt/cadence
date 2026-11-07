"""Regression for phase payloads overwritten by continuation checkpoints."""

from importlib import import_module
from pathlib import Path

import numpy as np

from cadence import Brain, LearnerConfig


def test_qualified_continuation_keeps_phase_and_checkpoint_bytes(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "benchmarks/acquisition"))
    benchmark = import_module("continual")
    config = LearnerConfig(
        beta=0.1,
        eta=0.1,
        eta_bias=0.01,
        free_steps=4096,
        nudged_steps=4096,
        tolerance=0.003,
        qualified=True,
        damping=3,
    )
    brain = Brain.compose(2, 2, modules=(2,), seed=3, learning=config, recall=False)
    inputs, labels = np.array([[1.0, 0.0]]), np.array([0])
    _, report = brain.learner.step(brain.stimulus(inputs, memory=False), labels)
    assert report["accepted"] == 1
    loaded = Brain.load(brain.save(tmp_path / "anchor.npz"))
    outcome = benchmark.graph_checkpoint_continuation(brain, loaded, inputs, labels, tmp_path)
    assert outcome["phase_payloads_retained"]
    assert outcome["accepted_equal"] and outcome["continued_arrays_equal"]
    for name, lesson in zip(("original", "loaded"), outcome["lessons"], strict=True):
        assert lesson["accepted"] == 1 and not lesson["failed"]
        phase_path = tmp_path / lesson["phase_file"]
        checkpoint_path = tmp_path / ("continued-" + name + ".npz")
        assert phase_path != checkpoint_path
        assert benchmark.sha256(phase_path) == lesson["phase_sha256"]
        with np.load(phase_path, allow_pickle=False) as phase:
            assert phase["free_v"].shape == (1, brain.connectome.n)
        assert Brain.load(checkpoint_path).learner.updates == brain.learner.updates
