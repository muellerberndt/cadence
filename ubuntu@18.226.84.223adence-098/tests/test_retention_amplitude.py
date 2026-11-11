"""Pure retained-panel contracts; no teaching, action, memory write or solve."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1] / "benchmarks/retention"
sys.path.insert(0, str(HERE))
SPEC = importlib.util.spec_from_file_location("amplitude_readout", HERE / "amplitude_readout.py")
amplitude = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(amplitude)


def reading(correct=9):
    return {
        "qualified_free": True,
        "refused": 0,
        "unrun": 0,
        "cues": [
            {
                "cue": q,
                "correct": correct,
                "prototype_correct": 1,
                "prototype_prediction": q % 2,
                "predictions": [q % 2] * 10,
                "margins": [0.5] * 10,
                "memory_values": [[1, 0]] * 10,
                "obsolete": 10 - correct if q < 2 else 0,
            }
            for q in range(3)
        ],
    }


def test_no_relaxed_partial_or_prototype_gate():
    assert amplitude.numeric_gate(reading())
    assert not amplitude.numeric_gate(reading(8))
    panel = reading(10)
    panel["cues"][0]["prototype_correct"] = 0
    assert not amplitude.numeric_gate(panel)
    panel = reading(10)
    panel["refused"] = 1
    assert not amplitude.numeric_gate(panel)


def test_baseline_requires_predictions_and_readout_parity():
    assert amplitude.same_baseline(reading(), reading())
    changed = reading()
    changed["cues"][0]["predictions"][3] = 1
    assert not amplitude.same_baseline(changed, reading())
    changed = reading()
    changed["cues"][0]["margins"][3] = 0.51
    assert not amplitude.same_baseline(changed, reading())


def test_incomplete_life_is_not_a_selected_endpoint():
    complete = {
        "complete": True,
        "seed": 0,
        "condition": "orthogonal",
        "exposure": 0,
        "endpoints": {name: {"intact": reading()} for name in amplitude.ENDPOINTS},
    }
    for endpoint in complete["endpoints"].values():
        endpoint["intact"]["world"] = 0
    incomplete = {"complete": False, "seed": 0}
    rows = amplitude.panels({"lives": [complete] * 4 + [incomplete] * 2})
    assert len(rows) == 40
    assert {row["amplitude"] for row in rows} == {1, 2}
    with pytest.raises(ValueError):
        amplitude.panels({"lives": [complete] * 3 + [incomplete] * 3})


def test_only_declared_gene_can_change_complete_checkpoint(tmp_path):
    original, changed = tmp_path / "original.npz", tmp_path / "changed.npz"
    generic = {"hippocampus": {"amplitude": 1, "writes": 8}, "pending": False}
    np.savez(original, generic=json.dumps(generic), efficacy=np.array([0.4]))
    generic["hippocampus"]["amplitude"] = 2
    np.savez(changed, generic=json.dumps(generic), efficacy=np.array([0.4]))
    assert amplitude.only_amplitude_changed(original, changed, 2)
    np.savez(changed, generic=json.dumps(generic), efficacy=np.array([0.41]))
    assert not amplitude.only_amplitude_changed(original, changed, 2)
