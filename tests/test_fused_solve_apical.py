"""A free solve of a graph with an apical compartment runs in one compiled call
(``fused.fused_solve`` with the compartment): the sweeps, the checks and every bit of the
launches ``Brain._equilibrate_local`` makes one settle or one check at a time."""

from __future__ import annotations

import types

import numpy as np
import pytest

import cadence as cd
from cadence import brain as brain_module

pytest.importorskip("numba")
pytestmark = pytest.mark.skipif(not brain_module._FUSED, reason="CADENCE_FUSED=0 reads the NumPy loop alone")


def graph():
    brain = cd.Brain.compose(10, 4, modules=(32, 16), apical=0.25, teaching="deep", backend="cpu",
                             seed=3)
    return brain.learner.brain


@pytest.mark.parametrize("rows", [1, 3])
def test_an_apical_free_solve_reads_the_bits_of_its_launches(rows: int) -> None:
    held = graph()
    assert held.apical is not None and held._solves_whole() and not held._solves_compiled()
    drive = np.random.default_rng(rows).normal(size=(rows, held.connectome.n)) * 0.7
    whole = held.equilibrate(drive, budget=300, chunk=16, tolerance=1e-9)
    warm = held.equilibrate(drive * 0.9, budget=300, chunk=16, tolerance=1e-9, state=whole.state)
    # a wrapped residual sees every launch: the solve runs one settle or check at a time
    held.residual = types.MethodType(type(held).residual, held)
    assert not held._solves_whole()
    launches = held.equilibrate(drive, budget=300, chunk=16, tolerance=1e-9)
    warm_launches = held.equilibrate(drive * 0.9, budget=300, chunk=16, tolerance=1e-9,
                                     state=launches.state)
    for a, b in ((whole, launches), (warm, warm_launches)):
        assert (a.steps, a.residual_checks) == (b.steps, b.residual_checks)
        for x, y in ((a.state.v, b.state.v), (a.state.activation, b.state.activation),
                     (a.residual, b.residual)):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
