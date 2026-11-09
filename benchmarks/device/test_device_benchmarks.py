"""The device benchmarks run, and the ledger counts a crossing where one is made (cadence#274)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ledger  # noqa: E402
import lives  # noqa: E402

# by its path under its own name: benchmarks/vocabulary holds a ``life`` module too, and one
# pytest process imports both
_spec = importlib.util.spec_from_file_location("device_life", HERE / "life.py")
assert _spec is not None and _spec.loader is not None
life = importlib.util.module_from_spec(_spec)
sys.modules["device_life"] = life
_spec.loader.exec_module(life)


def test_a_cpu_moment_has_no_crossing():
    row = ledger.read("chamber", "cpu", "step", 2, 1, 2, 0, True)
    assert row["seconds"] > 0 and row["crossings"] == []
    assert row["placement"]["backend"] == "cpu"


def test_the_ledger_names_the_site_of_a_crossing():
    torch = pytest.importorskip("torch")
    book = ledger.Ledger()
    tensor = torch.ones(3)

    def crossing():
        book.add("to host", int(tensor.numel() * tensor.element_size()), 0.001)

    book.active = True
    crossing()
    (way, site), (count, nbytes, seconds) = next(iter(book.rows.items()))
    assert way == "to host" and count == 1 and nbytes == 12 and seconds == 0.001
    assert site == "outside"  # no library frame made it


def test_live_follows_one_stream():
    with pytest.raises(ValueError, match="one stream"):
        ledger.read("chamber", "cpu", "live", 2, 0, 1, 0, False)


def test_a_life_and_lives_run_on_the_cpu():
    row = life.life("chamber", "cpu", 2, 3, False, 0)
    assert row["placed"] == "cpu" and row["seconds"] >= row["composed"] + row["first"]
    out = lives.serial(2, "cpu", 2, 1)
    assert out["per_life_moment"] > 0
    out = lives.streams(2, "cpu", 2, 1)
    assert out["placement"] == "cpu" and np.isfinite(out["per_life_moment"])
