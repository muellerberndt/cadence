"""The exposure-parity instrument (cadence#270): its orders, its comparison on synthetic rows
and a short life of passes with the rehearsal."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


gr = _load("grammar")
ex = _load("exposure")


def test_later_passes_take_the_transformer_epochs_order():
    passes = ex.order(5, 3, 10)
    assert list(passes[0]) == list(range(10))
    epochs = np.random.default_rng(np.random.SeedSequence([5, 99, 0]))
    assert list(passes[1]) == list(epochs.permutation(10))
    assert list(passes[2]) == list(epochs.permutation(10))


def _panels(level: float) -> dict:
    return {panel: {key: level} for panel, key in ex.READINGS.values()}


def test_the_comparison_picks_the_pass_within_the_controls_compute():
    life = {"arm": "rehearsal", "founder": 1, "passes": [
        {"pass": 1, "macs": 1.0, "readings": ex.scores(_panels(0.90))},
        {"pass": 2, "macs": 2.0, "readings": ex.scores(_panels(0.97))},
        {"pass": 3, "macs": 3.0, "readings": ex.scores(_panels(0.99))}]}
    control = {"mode": "offline", "founder": 1,
               "readings": [{"panels": _panels(1.0), "work": {"macs": 2.5}}]}
    out = ex.compare([life], [control], "rehearsal")
    assert out["E1 equal compute"]["passes"] == {1: 2}
    assert out["E1 equal compute"]["pass"] and out["E2 saturation"]["pass"]
    control["readings"][0]["work"]["macs"] = 1.5
    out = ex.compare([life], [control], "rehearsal")
    assert not out["E1 equal compute"]["pass"]
    assert set(out["E1 equal compute"]["apart"]) == set(ex.READINGS)


def test_a_life_of_passes_counts_its_rehearsal():
    row = ex.run_life({"arm": "rehearsal", "founder": 0, "passes": 2, "batches": 2,
                       "panel_batches": 1})
    assert [p["pass"] for p in row["passes"]] in ([1], [1, 2])
    work = row["passes"][0]["work"]
    assert work["rehearsal_row_sweeps"] > 0 and work["macs"] > 0
    assert row["passes"][-1]["macs"] >= work["macs"]
    assert "located" in row["passes"][-1]
