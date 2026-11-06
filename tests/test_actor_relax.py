"""The actor's readout relaxation (``actor_relax``, founder 0.0): at every actor update an output
whose activation at the scored act sits on the flat top of its range lowers its bias by the rate
times how far past ``1 - RELAX_BAND`` it sits; an output below that, and the founder, are left as
they were."""

from __future__ import annotations

import copy

import numpy as np
import pytest

import cadence as cd
from cadence.plasticity import RELAX_BAND

X = np.eye(4)


def life(brain, moments=40):
    action = brain.live(X[[0]])
    for t in range(moments):
        action = brain.live(X[[t % 4]], reward=[float(action[0] == t % 2)])
    return brain


def composed(**options):
    return cd.Brain.compose(4, 2, modules=(8,), seed=0, arousal=True, **options)


def test_the_founder_and_relax_zero_are_the_same_life():
    founder, zero = life(composed()), life(composed(actor_relax=0.0))
    np.testing.assert_array_equal(founder.brain.efficacy, zero.brain.efficacy)
    np.testing.assert_array_equal(founder.brain.bias, zero.brain.bias)
    assert founder.describe() == zero.describe()
    assert "actor_relax" not in founder.describe()["genes"]
    assert "relax" not in founder.basal_ganglia.config.to_dict()


def test_a_saturated_output_relaxes_its_bias_and_an_unsaturated_one_does_not():
    brains = {name: composed(actor_relax=relax) for name, relax in (("off", 0.0), ("on", 0.5))}
    motor = np.asarray(brains["off"].motor_index)
    for brain in brains.values():  # output 0 pinned near the top of its range
        bias = np.array(brain.brain.bias, copy=True)
        bias[motor[0]] += 20.0
        brain.learner.brain = brain.learner.brain.with_parameters(
            efficacy=np.array(brain.brain.efficacy), bias=bias)
    before = {name: np.array(b.brain.bias, copy=True) for name, b in brains.items()}
    acts = {}
    for name, brain in brains.items():
        brain.live(X[[0]])
        acts[name] = np.atleast_2d(np.asarray(brain.basal_ganglia._free.activation))[0, motor]
        brain.live(X[[1]], reward=[0.0])
    np.testing.assert_array_equal(acts["off"], acts["on"])  # the same act was scored
    assert acts["on"][0] > 1.0 - RELAX_BAND > acts["on"][1]
    moved = {name: np.asarray(b.brain.bias)[motor] - before[name][motor]
             for name, b in brains.items()}
    expected = -0.5 * (acts["on"][0] - (1.0 - RELAX_BAND))
    assert moved["on"][0] - moved["off"][0] == pytest.approx(expected, abs=1e-12)
    assert moved["on"][1] == pytest.approx(moved["off"][1], abs=1e-12)  # below the band: as off


def test_relax_continues_exactly_through_a_checkpoint_and_is_a_runtime_setting(tmp_path):
    brain = life(composed(actor_relax=0.05), moments=20)
    twin = cd.Brain.load(brain.save(tmp_path / "relax.npz"))
    assert twin.basal_ganglia.config.relax == 0.05
    for b in (brain, twin):
        life(b, moments=10)
    np.testing.assert_array_equal(twin.brain.bias, brain.brain.bias)
    np.testing.assert_array_equal(twin.brain.efficacy, brain.brain.efficacy)
    brain.retune(actor_relax=0.2)
    assert brain.basal_ganglia.config.relax == 0.2
    with pytest.raises(ValueError, match="relax"):
        composed(actor_relax=-1.0)


def test_grow_and_graft_keep_relax_and_it_adds_no_parameters():
    brain = life(composed(actor_relax=0.05), moments=10)
    assert brain.parameters() == life(composed(), moments=10).parameters()
    grown = brain.grow(history_decays=(0.5,))
    assert grown.basal_ganglia.config.relax == 0.05
    donor = composed(actor_relax=0.0)
    grafted = copy.deepcopy(brain).graft(donor, ["association", "motor"])
    assert grafted.basal_ganglia.config.relax == 0.05
