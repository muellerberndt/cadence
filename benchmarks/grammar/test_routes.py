"""Routes with a goal (cadence#257): the graphs, the held-out pairs, the scorer, a short life
of each speaker and a short control."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rt = _load("routes")


def solve(row) -> list[int]:
    """The route from the heard facts and the meaning alone."""
    edges, start, goal = [], None, None
    facts = row.pre[:-2]
    for t in range(0, len(facts), 4):
        edges.append((int(np.argmax(facts[t])), int(np.argmax(facts[t + 2]))))
    for x in row.pre[-2:]:
        place = int(np.argmax(x[rt.C["DOG"]:rt.C["DOG"] + len(rt.PLACES)]))
        if x[rt.C["FROM"]]:
            start = place
        if x[rt.C["TO"]]:
            goal = place
    nxt: dict[int, list[int]] = {}
    for a, b in edges:
        nxt.setdefault(a, []).append(b)
    for first in nxt[start]:
        route = [start, first]
        while route[-1] in nxt:
            route.append(nxt[route[-1]][0])
        if route[-1] == goal:
            return [w for place in route for w in (place, rt.NEAR)][:-1] + [rt.END]
    raise AssertionError("no branch reaches the goal")


def test_one_first_hop_reaches_the_goal_and_the_heard_moments_answer_it():
    rng = np.random.default_rng(0)
    split = rt.Split.of(0)
    for hops in rt.HOPS:
        for _ in range(50):
            row = rt.draw(rng, split, hops, held=False)
            assert len(row.target) == 2 * hops + 2 and row.target[-2] == row.meta["goal"]
            assert row.meta["first"] != row.meta["other"]
            assert len({a for a, _ in row.edges} | {b for _, b in row.edges}) == 2 * hops + 1
            assert solve(row) == row.target


def test_held_pairs_never_reach_training_and_fill_the_held_panel():
    split = rt.Split.of(3)
    assert len(split.held) == 8
    for rows in rt.train_batches(3, 40):
        assert all((r.meta["start"], r.meta["goal"]) not in split.held for r in rows)
    tests = rt.panels(3, 2)
    assert all((r.meta["start"], r.meta["goal"]) in split.held
               for b in tests["held"] for r in b)
    assert [b[0].meta["hops"] for b in tests["trained"]] == [2, 3]


def test_the_scorer():
    row = rt.episode(np.random.default_rng(1), 2, 0, 1)
    assert rt.score(row, row.target) == {"exact": True, "first_hop": True, "other_hop": False,
                                         "valid": True}
    wrong = [0, rt.NEAR, row.meta["other"], rt.NEAR, 1, rt.END]
    s = rt.score(row, wrong)
    assert not s["exact"] and s["other_hop"] and not s["valid"]


@pytest.mark.parametrize("speaker", ["word", "sliding"])
def test_a_short_life_of_each_speaker(speaker):
    row = rt.run_life({"speaker": speaker, "founder": 0, "batches": 2, "panel_batches": 1})
    assert row["k"] == (3 if speaker == "sliding" else 1)
    reading = row["readings"][-1]
    assert reading["work"]["lessons"] == 6 + 8  # routes of two and three hops with near, end
    assert set(reading["panels"]) == {"trained", "held"}


def test_a_short_control_and_the_gate():
    pytest.importorskip("torch")
    row = rt.control_life({"model": "lstm", "mode": "online", "founder": 0, "batches": 2,
                           "parameters": 5000, "panel_batches": 1})
    assert 0 <= row["panels"]["held"]["first_hop"] <= 1
    brains = [{"speaker": s, "founder": f, "readings": [{"panels": {"held": {
        "first_hop": 0.5 + (0.2 if s == "sliding" else 0.0) + 0.01 * f}}}]}
        for s in ("word", "sliding") for f in range(6)]
    g = rt.gate(brains)
    assert g["pays"] and abs(g["mean"] - 0.2) < 1e-9
