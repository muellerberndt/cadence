"""The lesson-step instrument (cadence#258): its plans, its readings and gates on synthetic
rows, and a short life of a candidate."""

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
ls = _load("lesson_step")


def _row(tag: str, founder: int, clause: list[float], level: float = 0.9) -> dict:
    """A life whose exact clauses at 3,000, 3,500 and 4,000 read ``clause``."""
    readings = []
    for b in range(500, 4001, 500):
        c = clause[[3000, 3500, 4000].index(b)] if b >= 3000 else level
        panels = {"clause": {"exact": c}, "dyck": {"exact": level},
                  "path": {"exact": level, "conditions": {"hops2": level}}}
        if b == 4000:
            panels.update({"clause_held": {"exact": level}, "roles": {"both": level}})
        readings.append({"phase": 0, "batches": b, "panels": panels})
    return {"tag": tag, "founder": founder, "readings": readings}


def test_the_plan_covers_every_arm_width_and_founder():
    specs = ls.plan("dev")
    assert len(specs) == len(ls.ARMS) * 2 * 3
    assert {s["module"] for s in specs} == {64, 128}
    assert all(s["arm"] == "default" and s["checkpoints"] == list(ls.CHECKPOINTS)
               for s in specs)
    assert gr.ARMS["default"] == {"recall": True}


def test_swing_and_levels_are_read_from_the_checkpoints():
    reading = ls.life(_row("dev/control/w64/f0", 0, [0.8, 0.9, 0.85]))
    assert np.isclose(reading["swing"], 0.1) and reading["clauses"] == 0.85
    assert reading["brackets"] == 0.9 and reading["two_hops"] == 0.9


def test_the_selection_takes_the_steadiest_eligible_candidate():
    rows = []
    for w in (64, 128):
        for f in (0, 1, 2):
            rows.append(_row(f"dev/control/w{w}/f{f}", f, [0.7, 0.9, 0.8]))
            rows.append(_row(f"dev/e99p5/w{w}/f{f}", f, [0.88, 0.9, 0.89]))
            rows.append(_row(f"dev/e999p5/w{w}/f{f}", f, [0.9, 0.9, 0.9], level=0.5))
    chosen = ls.select(ls.table(rows))
    assert chosen["chosen"] == "e99p5" and not chosen["flagged"]
    assert not chosen["eligible"]["e999p5"]


def test_the_gates_count_settled_founders():
    rows = []
    for w in (64, 128):
        for f in range(6):
            rows.append(_row(f"confirm/control/w{w}/f{f}", f, [0.7, 0.9, 0.8]))
            swing = [0.9, 0.9, 0.9] if f < 5 else [0.7, 0.9, 0.8]
            rows.append(_row(f"confirm/e99p5/w{w}/f{f}", f, swing))
    g = ls.gates(ls.table(rows), "e99p5")
    assert g["C1 settles, width 64"]["pass"] and g["C1 settles, width 64"]["within"] == "5 of 6"
    assert g["C2 levels hold, width 128"]["pass"]


def test_the_control_lives_with_its_tag():
    spec = {"arm": "default", "founder": 0, "phases": [["g1", 3]], "panel_batches": 1,
            "settings": ls.ARMS["control"], "tag": "smoke/control/w64/f0"}
    row = gr.run_life(spec)
    assert row["tag"] == spec["tag"] and row["settings"] == {"recall": True}
