"""The chamber's contracts: the dynamics, the demonstrator and the arms' senses."""

import cartpole
import numpy as np


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
    for arm, width in (("raw", 4), ("code", 64), ("code+dense", 104)):
        inputs, see = cartpole.senses(arm)
        assert inputs == width and see(state).shape == (1, width)
