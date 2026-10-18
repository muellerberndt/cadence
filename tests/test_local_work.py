"""The local-work instrument: block anatomy, the repair's contracts, determinism."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

import cadence as cd

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def lw():
    path = ROOT / "benchmarks" / "local_work" / "local_work.py"
    spec = importlib.util.spec_from_file_location("local_work", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["local_work"] = module
    spec.loader.exec_module(module)
    return module


def test_block_sparse_is_compose_at_one_skill(lw):
    dense = lw.build_brain("dense", (8,), 0).connectome
    sparse = lw.build_brain("sparse", (8,), 0).connectome
    assert np.array_equal(dense.pre, sparse.pre) and np.array_equal(dense.post, sparse.post)
    assert np.array_equal(dense.count, sparse.count) and np.allclose(dense.sign, sparse.sign)


def test_block_sparse_keeps_every_contact_inside_its_block(lw):
    widths = (8, 6, 10)
    brain = lw.build_brain("sparse", widths, 1)
    blocks = lw.block_of(brain, widths, 4, 2)
    c = brain.connectome
    assert (blocks[c.pre] == blocks[c.post]).all()
    for k, width in enumerate(widths):
        assert int((blocks == k).sum()) == 4 + 2 * width + 2
    dense = lw.build_brain("dense", widths, 1).connectome
    assert dense.n == c.n and dense.synapses > c.synapses
    # the block layout cuts the ranges at the block borders, one pair per projection block
    assert brain.brain.layout.ranges == 4 * len(widths)


def test_repair_leaves_an_unchanged_moment_alone_and_matches_the_library(lw):
    brain = lw.build_brain("sparse", (8, 8), 0)
    maps = lw.skill_maps(2, 0, 4, 2)
    lw.acquire(brain, maps, 0, d=4, lessons=20)
    brain.reset()
    x = lw.observations(np.zeros((1, 2), int), 4)
    brain.act(x, greedy=True)
    state = brain.basal_ganglia.state
    drive = brain.basal_ganglia._drive
    tolerance = float(brain.learner.config.tolerance)
    repair = lw.LocalRepair(brain.brain, state, drive, tolerance=tolerance)
    # the kept certificate stands under the same drive: no work, no complete check
    same, counts = repair.repair(drive)
    assert counts.rounds == 0 and counts.updates == 0 and counts.global_checks == 0
    assert np.array_equal(same.v, np.atleast_2d(state.v))
    # a changed cue: the repair qualifies and its complete residual is the library's
    x2 = lw.observations(np.array([[1, 0]]), 4)
    drive2 = brain.stimulus(x2)
    repaired, counts = repair.repair(drive2)
    assert counts.updates > 0 and counts.global_checks >= 1 and counts.qualified
    library = float(brain.brain.residual(drive2, repaired)[0])
    assert library <= tolerance
    assert abs(library - counts.residual) <= 1e-9
    # every contact pushed left a worklist member, and the cone is what was updated
    assert counts.messages <= counts.updates * int(np.max(repair.g.out_degree))
    assert 0 < counts.cone <= brain.connectome.n


def test_sync_reference_follows_the_library_sweep(lw):
    brain = lw.build_brain("dense", (8, 8), 2)
    x = lw.observations(np.array([[2, 1]]), 4)
    drive = brain.stimulus(x)
    warm = brain.brain.settle_batch(drive, steps=3)
    g = lw.Graph(brain.brain)
    # one synchronous sweep of the reference is one library step from the same state
    reference = lw.sync_reference(g, warm, drive, tolerance=0.0, max_sweeps=1)
    library = brain.brain.settle_batch(drive, steps=1, state=warm)
    assert reference["sweeps"] == 1
    np.testing.assert_allclose(reference["activation"], library.activation[0], atol=1e-12)


def test_case_is_deterministic_and_meets_its_gates(lw):
    spec = lw.CaseSpec("sparse", (8, 8), 0, lessons=40, probes=1, relation_probes=1, hold=12)
    first = lw.run_case(spec)
    second = lw.run_case(spec)

    def strip(case):
        moments = []
        for m in case["moments"]:
            m = json.loads(json.dumps(m))
            for arm in ("full", "local"):
                m[arm].pop("seconds", None)
            moments.append(m)
        return moments

    assert strip(first) == strip(second)
    gates = first["gates"]
    assert gates["same_action"] and gates["local_qualified"] and gates["state_within_gate"]
    kinds = [m["kind"] for m in first["moments"]]
    assert kinds[0] == "first" and "input" in kinds and "relation" in kinds
    change = next(m for m in first["moments"] if m["kind"] == "input")
    assert change["local"]["seed_changed_drive"] == 2  # the old and the new cue neuron
    assert change["local"]["updates"] < change["full"]["updates"]
    relation = next(m for m in first["moments"] if m["kind"] == "relation")
    assert relation["local"]["relation_changed_synapses"] > 0
    summary = lw.summarize(first)
    assert summary["by_kind"]["input"]["change"]["moments"] == 1
    assert isinstance(cd.__version__, str)
