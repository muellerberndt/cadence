"""The device instrument of the read-speech recipe (cadence#314, R3) on a small life."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_here = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("letters_device", _here / "device.py")
assert _spec is not None and _spec.loader is not None
dv = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("letters_device", dv)
_spec.loader.exec_module(dv)


def _run(capsys: pytest.CaptureFixture[str], *argv: str) -> list[dict]:
    old = sys.argv
    sys.argv = ["device.py", *argv]
    try:
        dv.main()
    finally:
        sys.argv = old
    return [json.loads(line) for line in capsys.readouterr().out.splitlines()]


def test_the_composition_is_the_recipes(monkeypatch: pytest.MonkeyPatch) -> None:
    assert dv.SHAPES["letters"] == ((40, 24), 29)
    assert dv.GENES["fields_motor"] == 1.0 and dv.GENES["learning_surprise"] is False
    x, labels = dv.data((40, 24), 29, 3, 2)
    assert x.shape == (2, 3, 960) and labels.shape == (2, 3) and labels.max() < 29
    assert np.abs(x).max() <= 0.9 + 1e-12


def test_a_short_life_on_a_second_place_matches_and_carries(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    pytest.importorskip("torch")
    rows = _run(capsys, "match", "--rows", "2", "--steps", "2", "--places", "cpu",
                "torch:cpu:float64", "--keep", str(tmp_path))
    assert [r.get("error") for r in rows] == [None, None]
    cpu, torch = rows
    assert cpu["saved_loads_on_cpu_to_the_bit"] and torch["saved_loads_on_cpu_to_the_bit"]
    assert torch["saved_loads_on_backend_to_the_bit"]
    assert torch["max_diff_efficacy_over_largest_step"] < 0.05
    assert torch["max_diff_dictionary"] == 0.0 and torch["greedy_choices_agree"] == 1.0
    saved = tmp_path / "letters-torch_cpu_float64.npz"
    carried = _run(capsys, "carry", "--saved", str(saved), "--rows", "2", "--steps", "1",
                   "--places", "cpu", "torch:cpu:float64")
    assert all(r["loads_to_the_saved_parameters"] for r in carried)
    assert carried[1]["max_diff_efficacy_over_largest_step"] < 0.05


def test_the_timing_reads_seconds_per_hour_of_audio(capsys: pytest.CaptureFixture[str]) -> None:
    (row,) = _run(capsys, "time", "--batches", "2", "--warm", "1", "--steps", "1")
    assert row["backend"] == "cpu" and "error" not in row
    step = row["lesson_seconds_per_step"] + row["act_seconds_per_step"]
    assert row["training_seconds_per_audio_hour"] == pytest.approx(step / (2 * 0.04) * 3600)
