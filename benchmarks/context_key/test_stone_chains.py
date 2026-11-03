"""The Stone Age chains of cadence#202 (Will Glynn's chamber, ported with credit): what each
chain shows, a short life per arm, and the observation hold at a retry."""

from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "stone_chains", Path(__file__).with_name("stone_chains.py")
)
assert spec is not None and spec.loader is not None
chains = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chains)
ARMS, CHAINS, Chamber, compose, life, mcnemar = (
    chains.ARMS, chains.CHAINS, chains.Chamber, chains.compose, chains.life, chains.mcnemar
)


def test_what_each_chain_shows():
    for chain, (first, second, third, fourth) in CHAINS.items():
        if chain == "ambiguous":  # the stages alias in pairs
            assert first == third and second == fourth and first != second
        elif chain == "invisible":  # the right wood changes nothing visible
            assert first == second and len({first, third, fourth}) == 3
        else:
            assert len({first, second, third, fourth}) == 4


def test_the_chamber_pays_a_finished_chain_and_charges_a_miss_only_with_a_cost():
    world = Chamber("ambiguous")
    assert world.execute(4) == (False, 0.0)
    for right in (0, 1, 2):
        assert world.execute(right) == (True, 0.0)
    assert world.execute(3) == (True, 2.0) and world.done and world.success
    costly = Chamber("ambiguous", cost=0.05)
    assert costly.execute(4) == (False, -0.05)
    for _ in range(8):
        costly.execute(4)
    assert costly.execute(4) == (False, -1.0) and costly.done and not costly.success


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_a_short_life_reads_every_arm(arm):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        row = life(("ambiguous", arm, "corrective", 0, 3, 2))
    assert row["arm"] == arm and 0.0 <= row["sampled"] <= 1.0 and len(row["first_try"]) == 4
    assert row["decisions"] >= 3


def test_a_retry_on_an_unchanged_observation_is_keyed_as_the_first_try():
    """Will Glynn's EXP-56 check, on the library's hold."""
    for hold, same in ((None, False), ("unchanged", True)):
        brain = compose("unchanged" if hold else "founder", 202_999)
        records = brain.memories["episodic"]
        x = Chamber("ambiguous").x()
        keys = []
        for _ in range(3):
            keys.append(records.key(brain.observation_code(x)))
            brain.act(x, greedy=True)
        assert all(np.array_equal(keys[0], k) for k in keys[1:]) == same


def test_the_exact_mcnemar_test():
    assert mcnemar(0, 0) == 1.0
    assert mcnemar(36, 2) < 1e-4
    assert mcnemar(5, 5) == 1.0
