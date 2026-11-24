"""The phase a moment settled is the phase its consumer reads. The composed brain hands the
actor the free phase it qualified, so the actor settles nothing of its own, on every backend
and for many lives at once. An imagined moment reads the code phase its branch prepared for
it; only an independent read (``predict``, ``stimulus(memory=False)``) settles a learned code
from rest."""

from __future__ import annotations

import numpy as np
import pytest
from places import PLACES, place

import cadence as cd
from cadence.plasticity import ActorCritic


@pytest.fixture
def settles(monkeypatch):
    """The actor's own settles, counted: the rows of each."""
    seen: list[int] = []
    settle = ActorCritic.settle

    def counted(self, drive):
        seen.append(len(drive))
        return settle(self, drive)

    monkeypatch.setattr(ActorCritic, "settle", counted)
    return seen


@pytest.mark.parametrize("name", PLACES)
def test_the_actor_reads_the_phase_the_brain_settled(name, settles):
    brain = cd.Brain.compose(12, 3, modules=(10,), seed=2, **place(name))
    rng = np.random.default_rng(0)
    brain.step(rng.random((2, 12)))
    for _ in range(3):
        brain.step(rng.random((2, 12)), reward=rng.random(2))
    brain.act(rng.random((2, 12)), greedy=True)
    assert settles == []


@pytest.mark.parametrize("name", [name for name in PLACES if "cpu-" in name or "cuda" in name])
def test_many_lives_settle_their_actor_once_a_moment(name, settles):
    lives = cd.Lives(12, 3, 4, modules=(10,), seeds=[1, 2, 3, 4], arousal=True, **place(name))
    rng = np.random.default_rng(0)
    lives.live(rng.random((4, 12)))
    for _ in range(3):
        lives.live(rng.random((4, 12)), reward=rng.random(4) - 0.3)
    assert settles == []
    stepped = cd.Lives(12, 3, 4, modules=(10,), seeds=[1, 2, 3, 4], **place(name))
    stepped.step(rng.random((4, 12)))
    stepped.step(rng.random((4, 12)), reward=rng.random(4))
    assert settles == []


def test_an_imagined_moment_reads_the_code_phase_its_branch_prepared(monkeypatch):
    brain = cd.Brain.compose(16, 4, modules=(12,), learned_code=24, seed=3, backend="cpu")
    rng = np.random.default_rng(5)
    brain.step(rng.random((2, 16)))
    for _ in range(4):
        brain.step(rng.random((2, 16)), reward=rng.random(2))
    solves, read = [], []
    solve, stimulus = brain._settle_code, brain._stimulus

    def counted(*args, **options):
        solves.append(bool(options.get("cold", False)))
        return solve(*args, **options)

    def kept(x, code, **options):
        coded = brain._coded_moment
        read.append(coded is not None and coded[0] is x and code is coded[1])
        return stimulus(x, code, **options)

    monkeypatch.setattr(brain, "_settle_code", counted)
    monkeypatch.setattr(brain, "_stimulus", kept)
    sequence = [rng.random((2, 16)) for _ in range(3)]
    phases = brain.imagine(sequence, budget=512)
    assert all(np.all(phase.qualified) for phase in phases)
    # a moment in the branch's context and the same moment without it: two solves, both warm
    assert solves == [False] * 6
    assert read == [True] * 3
    solves.clear()
    brain.predict(sequence[0])  # an independent read still settles the code from rest
    assert solves and all(solves)
