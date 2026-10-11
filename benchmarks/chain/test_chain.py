"""The prerequisite-chain nursery: its world, its controls, its receipts."""

import gzip
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("prerequisite_chain", HERE / "prerequisite_chain.py")
assert spec is not None and spec.loader is not None
chamber = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chamber)


@pytest.fixture(scope="module")
def protocol():
    protocol, _ = chamber.load_protocol(chamber.PROTOCOL)
    return protocol


def test_the_world_is_a_permuted_chain_with_a_fixed_background():
    world = chamber.make_world(3, 4)
    assert sorted(world["chain"] + world["distractors"]) == list(range(6))
    assert len(set(world["chain"])) == 4 and all(b > 4 for b in world["background"])
    assert chamber.make_world(3, 4) == world  # the same seed, the same world
    assert (
        chamber.make_world(4, 4)["chain"] != world["chain"]
        or chamber.make_world(4, 4)["background"] != world["background"]
    )


def test_observations_show_the_stage_only_when_observed():
    world = chamber.make_world(0, 4)
    seen = chamber.observe(world, 2, "observed")
    hidden = chamber.observe(world, 2, "hidden")
    assert seen.shape == (1, chamber.INPUTS) and seen[0, 2] == 1 and seen[0, 4] == 1
    assert hidden[0, 2] == 0 and hidden[0, 4] == 1
    for stage in range(4):
        assert np.array_equal(chamber.observe(world, stage, "hidden"), hidden)
        other = chamber.observe(world, stage, "observed")
        assert (other != seen).any() or stage == 2


def test_outcomes_advance_pay_and_restart_as_declared():
    world = chamber.make_world(0, 4)
    rng = np.random.default_rng(0)
    wrong = world["distractors"][0]
    assert chamber.outcome(world, 0, wrong, "end", rng) == (0.0, 0, False, False)
    for stage in range(3):
        reward, nxt, right, done = chamber.outcome(world, stage, world["chain"][stage], "end", rng)
        assert (reward, nxt, right, done) == (0.0, stage + 1, True, False)
        reward, nxt, right, done = chamber.outcome(
            world, stage, world["chain"][stage], "shaped", rng
        )
        assert reward == pytest.approx(0.25) and nxt == stage + 1
    reward, nxt, right, done = chamber.outcome(world, 3, world["chain"][3], "end", rng)
    assert (reward, nxt, right, done) == (1.0, 0, True, True)
    starts = {chamber.outcome(world, 3, world["chain"][3], "lessons", rng)[1] for _ in range(200)}
    assert starts == {0, 1, 2, 3}  # the backward lessons begin at every stage
    assert chamber.competent_income(4) == pytest.approx(0.25)


def test_the_stationary_ceiling_and_random_complete_at_their_rates():
    world = chamber.make_world(1, 4)
    rng = np.random.default_rng(1)
    chain = world["chain"]
    stationary = chamber.run_policy(world, "hidden", 20000, lambda x, s: chain[rng.integers(4)], 1)
    random = chamber.run_policy(world, "hidden", 20000, lambda x, s: rng.integers(6), 1)
    # four moments per step on average: 62.5 completions per thousand; six for random: 41.7
    assert 50 < stationary["completions_per_1000"] < 75
    assert 30 < random["completions_per_1000"] < 55
    assert stationary["one_action_share"] < 0.4


def test_brain_arms_change_only_what_they_declare(protocol):
    world = chamber.make_world(0, 4)
    need = protocol["need_share"] * chamber.competent_income(4)
    gene = chamber.make_brain(
        chamber.arm_point(protocol["brain"], "gene", protocol["stages"]), 0, world, need
    )
    small = chamber.make_brain(
        chamber.arm_point(protocol["brain"], "small", protocol["stages"]), 0, world, need
    )
    homeo = chamber.make_brain(
        chamber.arm_point(protocol["brain"], "homeo", protocol["stages"]), 0, world, need
    )
    assert gene.describe()["actor"]["eta"] == 0.03 and small.describe()["actor"]["eta"] == 0.003
    assert gene.describe()["learning"]["homeostasis_rate"] == 0.0
    assert homeo.describe()["learning"]["homeostasis_rate"] == 0.1
    assert np.array_equal(gene.brain.efficacy, small.brain.efficacy)
    assert gene.describe()["arousal"]["need"] == pytest.approx(need)


def test_a_short_run_writes_a_verifiable_receipt(tmp_path):
    protocol, _ = chamber.load_protocol(chamber.PROTOCOL)
    protocol["moments"], protocol["probe_every"], protocol["probe_moments"] = 400, 200, 100
    results = chamber.run(
        protocol, ["gene", "random", "tabular"], ["observed"], ["end"], [0], 1, tmp_path / "probes"
    )
    assert len(results["rows"]) == 3
    brain_row = [r for r in results["rows"] if r["arm"] == "gene"][0]
    assert len(brain_row["probes"]) == 3 and brain_row["totals"]["moments"] == 400
    assert set(brain_row["final"]) >= {"greedy", "sampled", "stages", "greedy_stage_accuracy"}
    body = {"schema": chamber.SCHEMA, "protocol": protocol, "seeds": [0], **results}
    out = tmp_path / "receipt.json.gz"
    manifest = chamber.Receipt.build(chamber.SCHEMA, {}, chamber.sources()).source
    chamber.write_receipt(out, body, manifest)
    valid, reason = chamber.verify(out, current=True)
    assert valid, reason
    assert "prerequisite-chain/1" in chamber.markdown(chamber.read_receipt(out)["body"])
    tampered = json.loads(gzip.decompress(out.read_bytes()))
    tampered["body"]["rows"][0]["seed"] = 99
    bad = tmp_path / "tampered.json.gz"
    bad.write_bytes(gzip.compress(json.dumps(tampered).encode()))
    assert not chamber.verify(bad)[0]
