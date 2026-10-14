"""The night-replay chamber's contract: frozen readings, equal experience, reported shares."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _chamber():
    path = Path(__file__).resolve().parents[1] / "benchmarks/replay/night_replay.py"
    spec = importlib.util.spec_from_file_location("night_replay", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _checkpoint(brain, path):
    with np.load(brain.save(path), allow_pickle=False) as saved:
        return {name: saved[name].copy() for name in saved.files}


def test_readings_leave_the_live_brain_untouched(tmp_path):
    chamber = _chamber()
    obs, outcome, cue = chamber.make_day(0, 6, True)
    brain = chamber.make_brain(chamber.SETTINGS["trace-1-0.8+actor-0.1"], 0)
    chamber.live(brain, obs, outcome, cost=2.0, scale=1.0, owed=None)
    before = _checkpoint(brain, tmp_path / "before.npz")
    reading = chamber.readings(brain, obs, cue, True)
    after = _checkpoint(brain, tmp_path / "after.npz")
    assert before.keys() == after.keys()
    for name in before:
        np.testing.assert_array_equal(before[name], after[name], err_msg=name)
    assert sum(reading["greedy"].values()) == len(obs)
    assert 0.0 <= reading["dependence"] <= 1.0
    assert 0.0 <= reading["cue_agreement"] <= 1.0
    assert len(reading["mean_policy"]) == 3 and abs(sum(reading["mean_policy"]) - 1) < 1e-9
    assert len(reading["recall_drive"]) == 3


@pytest.mark.parametrize("world", ["signal", "noise"])
def test_the_night_and_the_awake_control_consume_the_same_experience(world):
    chamber = _chamber()
    result = chamber.run_one(world, "defaults", 1, days=8, rounds=1)
    assert result["night_decisions"] == result["awake_decisions"] == 8 + 8 + 7
    assert sum(result["day_sampled"].values()) == 8
    assert sum(result["night_sampled"].values()) == result["night_decisions"]
    for stage in ("before", "after_night", "after_awake"):
        assert 0.0 <= result[stage]["dependence"] <= 1.0
        assert ("cue_agreement" in result[stage]) == (world == "signal")
    # Outcomes of several units against the default cap of 1.0 clip to their sign.
    assert 0.0 <= result["day_capped"] <= 1.0 and 0.0 <= result["night_capped"] <= 1.0
    assert result["options"] == {}


def test_settings_change_only_what_they_name():
    chamber = _chamber()
    base = chamber.make_brain(chamber.SETTINGS["defaults"], 0)
    quiet = chamber.make_brain(chamber.SETTINGS["trace-1-0.8+actor-0.1"], 0)
    assert quiet.working_memory is not None and base.working_memory is not None
    assert (quiet.working_memory.amplitude, quiet.working_memory.decay) == (1.0, 0.8)
    assert (base.working_memory.amplitude, base.working_memory.decay) == (3.0, 0.0)
    assert quiet.basal_ganglia.config.eta == 0.1 and quiet.basal_ganglia.config.eta_bias == 0.01
    assert base.basal_ganglia.config == quiet.basal_ganglia.config.__class__(
        **{**quiet.basal_ganglia.config.to_dict(), "eta": 1.0, "eta_bias": 0.05}
    )
    assert chamber.make_brain(chamber.SETTINGS["no-memory"], 0).hippocampus is None
    assert chamber.make_brain(chamber.SETTINGS["trace-off"], 0).working_memory.amplitude == 0.0
    with pytest.raises(ValueError):
        chamber.run_one("dream", "defaults", 0, days=4, rounds=1)
