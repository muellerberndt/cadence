"""Contract tests of the long reward life chamber (cadence#296)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


def _load(name: str):
    """This chamber's module under a name of its own: other benchmarks have a chamber.py."""
    key = f"actor_latch_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, Path(__file__).with_name(f"{name}.py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


chamber = _load("chamber")


def episodes(world: chamber.Bandit, policy, count: int) -> list[tuple[int, float, str]]:
    out = []
    for _ in range(count):
        world.reset()
        moments, returned = 0, 0.0
        while True:
            transition = world.step(policy())
            moments += 1
            returned += transition.reward
            if transition.done:
                out.append((moments, returned, transition.outcome))
                break
    return out


def test_an_observation_is_the_retina_the_body_ports_and_the_previous_button() -> None:
    for kind in chamber.CHAMBERS.values():
        world = kind(width=12, seed=3)
        first = world.reset()
        assert first.shape == (world.inputs,) == (12 + 3 + 3,)
        assert not first[-3:].any()  # no previous button at an episode's first moment
        world.step(2)
        second = world.observe()
        assert np.array_equal(second[-3:], [0.0, 0.0, 1.0])
        assert ((second[:12] >= 0) & (second[:12] <= 1)).all()


def test_a_button_outside_the_set_is_refused_before_the_world_moves() -> None:
    for kind in chamber.CHAMBERS.values():
        world = kind(width=4, seed=0)
        world.reset()
        for wrong in (-1, 3):
            with pytest.raises(ValueError):
                world.step(wrong)
        assert world.moment == 0


def test_random_buttons_complete_the_bandit_as_random_buttons_complete_doom_basic() -> None:
    rng = np.random.default_rng(0)
    world = chamber.Bandit(width=4, seed=0)
    rows = episodes(world, lambda: int(rng.integers(3)), 600)
    completion = np.mean([outcome == "completion" for _, _, outcome in rows])
    assert 0.57 < completion < 0.69  # basic: random .63
    held = episodes(chamber.Bandit(width=4, seed=0), lambda: 0, 300)
    assert np.mean([outcome == "completion" for _, _, outcome in held]) > 0.9


def test_a_non_paying_button_held_in_the_survival_chamber_lives_out_its_health() -> None:
    world = chamber.Survival(width=4, seed=0, paying=0)
    rows = episodes(world, lambda: 1, 3)
    # the latched health-gathering founders of the Doom body: a return of -0.040
    for moments, returned, outcome in rows:
        assert (moments, outcome) == (97, "death")
        assert returned == pytest.approx(-0.04)
    held = episodes(chamber.Survival(width=4, seed=0, paying=0), lambda: 0, 60)
    rng = np.random.default_rng(1)
    random = episodes(chamber.Survival(width=4, seed=0, paying=0), lambda: int(rng.integers(3)), 60)
    assert np.mean([m for m, _, _ in held]) > 2 * np.mean([m for m, _, _ in random]) > 2 * 97


def test_a_brain_life_reads_by_quarters(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    life = chamber.live(8, 0, settings={"modules": [8]}, world={"width": 6, "length": 10})
    assert life["parameters"] > 0 and len(life["episodes"]) == 8
    rows = chamber.quarters(life)
    assert [q["quarter"] for q in rows] == [1, 2, 3, 4]
    for q in rows:
        assert sum(q["shares"]) == pytest.approx(1.0)
        assert q["saturation"] is not None and q["scale_step"] is not None
    path = tmp_path / "life.json"
    path.write_text(json.dumps(life))
    chamber.read([path])
    assert "| bandit: modules=[8] | 0 |" in capsys.readouterr().out


def test_a_projection_rate_multiplies_that_projection_alone() -> None:
    brain = chamber.compose(9, 3, 0, {"modules": [8]}, {"sensory->association": 0.5})
    rate = brain.learner.synapse_rate
    connectome = brain.connectome
    sensory = np.zeros(connectome.n, dtype=bool)
    sensory[np.asarray(connectome.populations["sensory"])] = True
    assert set(np.unique(rate[sensory[connectome.pre]])) == {0.5}
    assert set(np.unique(rate[~sensory[connectome.pre]])) == {1.0}
    with pytest.raises(ValueError):
        chamber.compose(9, 3, 0, {"modules": [8]}, {"motor->sensory": 0.5})


def test_the_controls_learn_nothing() -> None:
    frozen = chamber.live(3, 1, player="frozen", settings={"modules": [8]},
                          world={"width": 6, "length": 10})
    assert all(r["saturation"] is None for r in frozen["episodes"])
    random = chamber.live(3, 1, player="random", world={"width": 6, "length": 10})
    assert random["parameters"] == 0
