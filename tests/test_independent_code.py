"""An independent read of a learned code (``predict``, ``accuracy``, ``fit``,
``stimulus(memory=False)``, ``observation_code``) settles the region once, from rest, at the
step every state of the region is stable under. What it reads is a function of the tables and
the observations: a brain in the middle of a life reads what its file reads on a brain at rest.
The step and the sub-network at it are kept while the tables stand."""

from __future__ import annotations

import numpy as np
import pytest
from places import place

import cadence as cd

PLACES = ["numpy", "torch-cpu-float64"]


def lived(name: str, moments: int = 5) -> tuple[cd.Brain, np.random.Generator]:
    brain = cd.Brain.compose(16, 4, modules=(12,), learned_code=24, seed=3, **place(name))
    rng = np.random.default_rng(7)
    brain.step(rng.random((8, 16)))
    for _ in range(moments):
        brain.step(rng.random((8, 16)), reward=rng.random(8))
    return brain, rng


@pytest.mark.parametrize("name", PLACES)
def test_a_living_brain_reads_what_its_file_reads_at_rest(name, tmp_path):
    brain, rng = lived(name)
    query = rng.random((8, 16))  # as many rows as the stream holds: a warm start would fit
    for moment in range(3):
        fresh = cd.Brain.load(brain.save(tmp_path / f"{moment}.npz"), **place(name))
        fresh.reset()  # the same parameters, no stream
        np.testing.assert_array_equal(brain.stimulus(query, memory=False),
                                      fresh.stimulus(query, memory=False))
        np.testing.assert_array_equal(brain.observation_code(query),
                                      fresh.observation_code(query))
        np.testing.assert_array_equal(brain.predict(query), fresh.predict(query))
        brain.step(rng.random((8, 16)), reward=rng.random(8))  # the tables move


def test_the_rest_step_and_its_graph_are_kept_while_the_tables_stand(monkeypatch):
    brain, rng = lived("numpy")
    region = brain.learned_code
    read = []
    bound = region.rest_stability
    monkeypatch.setattr(region, "rest_stability", lambda: read.append(1) or bound())
    live = region.halvings, region.graph, region.last
    query = rng.random((5, 16))
    brain.predict(query)
    kept = region._rest
    assert kept is not None and kept[0] is region.dictionary and kept[4] is not None
    brain.predict(rng.random((3, 16)))
    brain.accuracy(query, np.zeros(5, dtype=int))
    assert len(read) == 1 and region._rest is kept  # one eigenvalue read, one graph
    assert (region.halvings, region.graph, region.last) == live
    brain.step(rng.random((8, 16)), reward=rng.random(8))  # a learning step of the region
    brain.predict(query)
    assert len(read) == 2 and region._rest[0] is region.dictionary


def test_an_independent_read_takes_one_solve_of_the_code(monkeypatch):
    brain, rng = lived("numpy")
    solves = []
    solve = brain._settle_code

    def counted(*args, **options):
        solves.append((options.get("cold", False), options.get("report", True)))
        return solve(*args, **options)

    monkeypatch.setattr(brain, "_settle_code", counted)
    query = rng.random((6, 16))
    brain.predict(query)
    assert solves == [(True, False)]
    assert brain.last_settlement["code_steps"] > 0  # the read's own report
    solves.clear()
    brain.fit(query, np.arange(6) % 4, batch=3, epochs=2)
    assert solves[:4] == [(True, False)] * 4  # a solve a minibatch, then the scores'
