"""The forager chamber's world, teacher and readings, small."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import chamber as ch  # noqa: E402
import world as w  # noqa: E402


def test_the_senses_have_the_declared_layout_and_the_teacher_forages():
    world = w.Worlds(4, seed=0)
    sensed = world.senses()
    assert w.flat(sensed).shape == (4, 104)
    assert set(sensed) == set(w.SENSES) and set(sensed["food"]) == set(w.SENSES["food"])
    for _ in range(200):
        world.step(w.teach(world))
    assert world.eaten.sum() > 4                     # the teacher eats
    still = w.Worlds(4, seed=0)
    for _ in range(200):
        still.step(np.full(4, w.REST))
    assert still.eaten.sum() == 0                    # a body at rest eats nothing


def test_a_bite_eats_only_a_plant_in_reach_ahead():
    world = w.Worlds(1, seed=3)
    world.pose[0] = [6.0, 6.0, 0.0]                 # the middle of the ground, facing east
    world.rocks[0] = [[1.0, 1.0]] * w.ROCKS          # the rocks in a corner
    world.plants[0] = [[11.0, 11.0]] * w.PLANTS      # every plant out of reach
    world.plants[0, 0] = [6.5, 6.0]                  # one plant just ahead
    assert w.teach(world)[0] == w.BITE
    assert world.step(np.array([w.BITE]))[0] == 1 and world.eaten[0] == 1
    assert not np.allclose(world.plants[0, 0], [6.5, 6.0])   # it grows again elsewhere
    world.plants[0, 0] = [5.5, 6.0]                  # a plant behind the body
    assert world.step(np.array([w.BITE]))[0] == 0    # out of the bite's cone
    assert w.teach(world)[0] in (w.LEFT, w.RIGHT)    # the teacher turns to it


def test_a_short_life_reads_every_arm():
    for arm in ("code", "both"):
        brain = ch.compose(0, arm)
        newborn = ch.compose(0, arm)
        lessons = ch.Lessons(brain, 0, bodies=4)
        taught = lessons.next(12)
        assert 0.0 <= taught["agreement"] <= 1.0
        own = ch.own_path(brain, 0, moments=10, bodies=4)
        path = ch.teacher_path(brain, newborn, 0, moments=30, bodies=4)
        assert set(own) == {"eaten", "agreement", "by_action"}
        assert set(path["readout"]) >= {"observation", "code", "state", "newborn", "moments"}
    assert ch.compose(0, "both").describe()["layout"]["code"]["cortex"] == "both"
