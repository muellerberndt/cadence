"""The strength chamber (cadence#336) in short: its readings hold their shape and the law."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _load():
    key = "memory_strength_chamber"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, HERE / "strength.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


ch = _load()


def test_the_dose_curve_follows_the_law_and_main_keeps_every_record():
    curves, gone = ch.dose(0, "strength", patterns=3, spans=(0, 60, 1000))
    assert curves[0] == [1.0] * 5
    assert curves[60] == [0.0, 1.0, 1.0, 1.0, 1.0] and curves[1000] == [0.0, 0.0, 0.0, 1.0, 1.0]
    assert gone[:3] == [54.0, 100.0, 850.0] and not np.isfinite(gone[3:]).any()
    assert [round(ch.law(n)) for n in (1, 2, 4)] == [53, 100, 850]
    main, untouched = ch.dose(0, "main", patterns=3, spans=(0, 1000))
    assert main[1000] == [1.0] * 5 and not np.isfinite(untouched).any()


def test_one_surprising_write_is_kept_and_the_clean_form_beats_every_take():
    shot = ch.one_shot(1, "strength", patterns=3, span=1000)
    assert shot["one write, surprise 7"] == shot["told 8 times"] == 1.0
    assert shot["one write, surprise 0"] == 0.0
    form = ch.clean_form(2, "strength", patterns=3)
    assert all(form[n][0] > form[n][1] for n in (4, 8, 16))
    plain = ch.clean_form(2, "main", patterns=3)
    assert all(plain[n][0] <= plain[n][1] for n in (1, 2, 4, 8, 16))
