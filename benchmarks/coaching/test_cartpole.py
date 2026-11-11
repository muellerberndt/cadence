"""The chamber's contracts: the dynamics, the demonstrator and the arms' senses."""

import importlib.util
import sys
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "coaching_cartpole", Path(__file__).parent / "cartpole.py"
)
cartpole = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cartpole
spec.loader.exec_module(cartpole)


def test_the_demonstrator_balances_every_held_out_start():
    assert all(cartpole._demonstrate(seed) == cartpole.STEPS for seed in cartpole.GATE)


def test_a_constant_push_drops_the_pole_and_ends_the_episode():
    pole = cartpole.CartPole(0)
    while pole.step(1):
        pass
    fallen = abs(pole.state[0]) > cartpole.X_LIMIT or abs(pole.state[2]) > cartpole.ANGLE_LIMIT
    assert pole.steps < 100 and fallen


def test_the_arms_senses():
    state = np.array([0.1, -0.2, 0.03, 0.4])
    for arm, width in (("raw", 4), ("code", 64), ("code+dense", 104), ("percentile", 64),
                       ("percentile+clamp", 64)):
        inputs, see = cartpole.senses(arm)
        assert inputs == width and see(state).shape == (1, width)


def test_a_reading_past_a_percentile_range_goes_dark_only_unclamped():
    far = np.array([0.0, 0.0, 0.0, 2.0])  # the pole turning far faster than the demonstrator's
    faded = cartpole.senses("percentile")[1](far)
    clamped = cartpole.senses("percentile+clamp")[1](far)
    assert not faded[0, 48:].any() and clamped[0, 63] == 1.0
