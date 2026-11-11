"""The specific fact: two codes told apart, and the gain selected on a specific training fact."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

import cadence as cd

HERE = Path(__file__).resolve().parent


def _load(name: str):
    """The instrument module ``name`` of this folder, loaded once under its own module name."""
    key = "instruments_" + name
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, HERE / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


instrument = _load("protocol")
Levels, shared_code = instrument.Levels, instrument.shared_code


def two_codes() -> cd.Connectome:
    """Inputs A (0, 1) and B (2, 3); hidden cells 4, 5 answer A, 6, 7 answer B, 8 answers both."""
    pre = [0, 1, 0, 1, 2, 3, 2, 3, 0, 2]
    post = [4, 4, 5, 5, 6, 6, 7, 7, 8, 8]
    return cd.Connectome.from_synapses(
        9,
        pre=pre,
        post=post,
        count=[100] * len(pre),
        populations={"a": [0, 1], "b": [2, 3], "hidden": [4, 5, 6, 7, 8]},
    )


def test_shared_code_is_the_intersection_over_the_union() -> None:
    a = np.array([0, 0, 0, 0, 1, 1, 0, 0, 1], dtype=float)
    b = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=float)
    assert shared_code(a, b, [4, 5, 6, 7, 8], 0.5) == pytest.approx(0.2)
    assert shared_code(a, a, [4, 5, 6, 7, 8], 0.5) == pytest.approx(1.0)
    assert shared_code(0 * a, 0 * b, [4, 5], 0.5) == 0.0


def test_specific_needs_two_codes_that_barely_overlap() -> None:
    code = {"mean": 0.5, "fraction": 0.6}
    assert instrument.evaluate_predicate("specific", {**code, "shared": 0.2}, code)
    assert not instrument.evaluate_predicate("specific", {**code, "shared": 0.4}, code)
    # a silent or near-empty code cannot be specific (a dead control has none), and a reading
    # without a comparison cannot pass
    dead = {"mean": 0.0, "fraction": 0.001}
    assert not instrument.evaluate_predicate("specific", {**dead, "shared": 0.0}, code)
    assert not instrument.evaluate_predicate("specific", {**code, "shared": 0.0}, dead)
    assert not instrument.evaluate_predicate("specific", code, code)
    assert instrument.evaluate_predicate(
        "specific", {**code, "shared": 0.4}, code, Levels(specific_max=0.5)
    )


def test_protocol_scores_a_specific_row_and_selects_on_a_specific_fact() -> None:
    connectome = two_codes()
    protocol = instrument.Protocol(
        stimuli={"A": ("a",), "B": ("b",)},
        rows=[
            instrument.Row("A and B are told apart", "A", "hidden", "specific", versus="B"),
            instrument.Row("A is not told from itself", "A", "hidden", "specific", versus="A"),
        ],
        training=[("A", "hidden", "specific", "B")],
        steps=60,
    )
    brain = cd.NeuralGraph(connectome, cd.NeuronModel(gain=0.05))
    score = protocol.score(brain)
    told, same = score["rows"]
    assert told["passed"] and told["reading"]["shared"] == pytest.approx(0.2)
    assert not same["passed"] and same["reading"]["shared"] == pytest.approx(1.0)
    assert score["training"][0]["versus"] == "B" and score["training"][0]["passed"]
    # fraction counts the members in the code at code_level, the level specific uses
    graded = instrument.Protocol(
        stimuli=protocol.stimuli, rows=protocol.rows, levels=Levels(code_level=0.05)
    )
    assert graded.score(brain)["rows"][0]["reading"]["fraction"] >= told["reading"]["fraction"]
    # a gain at which nothing settles cannot pass the fact; the selection lands on one that can
    gain, table = instrument.select_gain(
        lambda g: cd.NeuralGraph(connectome, cd.NeuronModel(gain=g)),
        protocol,
        [0.001, 0.05, 0.1],
        sparsity_cap=None,
    )
    assert gain == 0.05
    assert [row["facts_passed"] for row in table] == [0, 1, 1]
    assert "A->hidden vs B" in table[1]["readings"]
