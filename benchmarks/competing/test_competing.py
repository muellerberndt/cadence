"""Guards of the competing-skill ring: its world, its arms, its readings and its receipts."""

import gzip
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

import cadence as cd

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("competing_skills", HERE / "competing_skills.py")
chamber = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chamber)


@pytest.fixture(scope="module")
def protocol():
    return json.loads((HERE / "protocol.json").read_text())


@pytest.fixture(scope="module")
def small_receipt(tmp_path_factory):
    """A receipt of a short run over three arms, as the chamber writes it."""
    path = tmp_path_factory.mktemp("receipt") / "small.json.gz"
    run = ["--arms", "gene", "frozen", "random", "--rings", "full", "--seeds", "0"]
    run += ["--nursery", "400", "--ring", "300", "--probe-every", "150", "--workers", "1"]
    checkpoints = tmp_path_factory.mktemp("nurseries")
    assert chamber.main([*run, "--out", str(path), "--checkpoints", str(checkpoints)]) == 0
    return path, json.loads(gzip.decompress(path.read_bytes()))["body"]


def test_the_protocol_declares_the_arena_founder_and_the_rings(protocol):
    assert protocol["schema"] == chamber.SCHEMA == "competing-skills/1"
    brain = protocol["brain"]
    assert brain["modules"] == [48, 24] and brain["sensory_scale"] == 4.0
    assert (brain["eta"], brain["eta_bias"], brain["lam"], brain["gamma"]) == (
        0.03,
        0.003,
        0.6,
        0.95,
    )
    assert brain["eta_critic"] == 5.0 and brain["temperature"] == 0.3
    assert set(protocol["stages"]) == {"small", "smallest", "calm", "stage"}
    assert set(protocol["world"]["rings"]) == set(chamber.RINGS)
    assert protocol["gates"] is None  # development: no gate is declared yet


def test_the_world_pays_the_right_action_and_the_twins_compete():
    for seed in range(4):
        world = chamber.make_world(seed)
        patterns, correct = world["patterns"], world["correct"]
        assert patterns.shape == (8, 16) and (patterns.sum(axis=1) == 4).all()
        for i in range(4):
            shared = patterns[i] @ patterns[4 + i]
            assert shared == 2, "a ring twin keeps two of its nursery twin's inputs"
            assert correct[i] != correct[4 + i], "and asks a different action"
        assert sorted(correct[:4]) == [0, 1, 2, 3] and sorted(correct[4:]) == [0, 1, 2, 3]
        assert chamber.reward_of(0, int(correct[0]), correct, False) == 1.0
        assert chamber.reward_of(0, int(correct[0]), correct, True) == 0.5
        assert chamber.reward_of(0, int((correct[0] + 1) % 4), correct, False) == 0.0


def test_the_rings_present_what_they_declare():
    situations, burns, coins = chamber.make_moments(
        0, 2000, competing=True, burn=True, burn_rate=0.25
    )
    assert situations.max() == 7 and 0.2 < burns.mean() < 0.3 and coins.shape == (2000,)
    situations, burns, _ = chamber.make_moments(
        0, 2000, competing=False, burn=False, burn_rate=0.25
    )
    assert situations.max() == 3 and not burns.any()
    again, _, same_coins = chamber.make_moments(0, 2000, competing=True, burn=False, burn_rate=0.25)
    with_burn = chamber.make_moments(0, 2000, competing=True, burn=True, burn_rate=0.25)
    assert np.array_equal(again, with_burn[0]) and np.array_equal(same_coins, with_burn[2]), (
        "burn changes neither the situations nor the coins"
    )


def test_a_soft_contingency_pays_at_its_stated_rates():
    correct = np.array([0, 1, 2, 3, 1, 2, 3, 0])
    soft = {"right": 0.6, "wrong": 0.3}
    coins = np.random.default_rng(0).random(20000)
    right = np.mean([chamber.reward_of(0, 0, correct, False, c, soft) for c in coins])
    wrong = np.mean([chamber.reward_of(0, 1, correct, False, c, soft) for c in coins])
    assert abs(right - 0.6) < 0.02 and abs(wrong - 0.3) < 0.02
    assert chamber.reward_of(0, 0, correct, True, 0.99, soft) == chamber.BURN
    assert chamber.expected_income(1.0, soft, 0.25) == pytest.approx(0.6 - 0.125)
    assert chamber.expected_income(0.25, chamber.HARD, 0.0) == 0.25


def test_the_arms_change_only_what_they_declare(protocol):
    stages = protocol["stages"]
    for arm in chamber.ARMS:
        brain = chamber.make_brain(protocol["brain"], 0)
        before = brain.describe()
        chamber.enter_ring(brain, arm, stages)
        after = brain.describe()
        actor, arousal = after["actor"], after["arousal"]
        if arm in ("gene", "frozen", "random"):
            assert after == before
        elif arm == "small":
            assert (actor["eta"], actor["eta_bias"]) == (0.003, 0.0003)
            assert arousal == before["arousal"]
        elif arm == "smallest":
            assert (actor["eta"], actor["eta_bias"]) == (0.001, 0.0001)
        elif arm == "calm":
            assert arousal["threshold"] == 0.5 and actor == before["actor"]
        elif arm == "stage":
            assert (actor["eta"], actor["eta_bias"]) == (0.001, 0.0001)
            assert arousal["need"] == 0.0 and arousal["heat"] == 0.0
            assert after["learning"]["temperature"] == 0.2
            assert brain.arousal is not None and brain.arousal.level == 0.0


def test_synapse_groups_partition_the_synapses_and_the_neurons(protocol):
    brain = chamber.make_brain(protocol["brain"], 0)
    groups = chamber.synapse_groups(brain)
    synapses = np.zeros(brain.connectome.synapses, dtype=int)
    neurons = np.zeros(brain.connectome.n, dtype=int)
    for name, mask in groups.items():
        if name.startswith("bias:"):
            neurons += mask
        else:
            synapses += mask
    assert (synapses == 1).all() and (neurons == 1).all()
    assert {
        "sensory->module_0",
        "association->motor",
        "motor->association",
        "prefrontal->association",
        "module_0->association",
    } <= set(groups)


def test_consistency_tells_a_signal_from_noise():
    tally = chamber.Consistency(3)
    for _ in range(50):
        tally.add(np.array([0.1, 0.0, 0.1]))
    for k in range(50):
        tally.add(np.array([0.1, 0.0, -0.1 if k % 2 else 0.1]))
    groups = {"all": np.ones(3, dtype=bool), "bias:x": np.ones(3, dtype=bool)}
    out = tally.by_group(groups)["all"]
    assert out["moved"] == 2 and out["synapses"] == 3
    # synapse 0 always pushed one way (1.0); synapse 2 pushed one way, then alternating (0.5)
    assert abs(out["consistency"] - (1.0 + 0.5) / 2) < 1e-9


def test_a_short_run_writes_a_verifiable_receipt_with_the_controls_in_place(small_receipt, capsys):
    path, body = small_receipt
    assert body["protocol"]["phases"] == {"nursery": 400, "ring": 300}
    assert body["overrides"] == {"nursery": 400, "ring": 300, "probe_every": 150}
    nursery = body["nurseries"][0]
    assert nursery["totals"]["moments"] == 400 and len(nursery["checkpoint_sha256"]) == 64
    rows = {row["arm"]: row for row in body["rows"]}
    assert set(rows) == {"gene", "frozen", "random"}
    gene = rows["gene"]
    assert gene["totals"]["moments"] == 300 and len(gene["probes"]) == 3
    assert gene["probes"][0]["moment"] == 0 and gene["probes"][-1]["moment"] == 300
    assert "association->motor" in gene["drift"] and "association->motor" in gene["consistency"]
    frozen = rows["frozen"]
    # a frozen brain's parameters do not change; its trace does. The reset-trace probe of
    # the same parameters therefore answers identically at the door and at the end, while
    # the carried probe may differ.
    door, end = frozen["probes"][0]["readings"], frozen["final"]
    assert door["reset"] == end["reset"]
    assert door["reset"]["accuracy_a"] == 1.0
    assert "reset" in gene["final"] and set(gene["final"]["reset"]) >= {"accuracy_a", "margin_a"}
    assert (
        frozen["probes"][0]["readings"]["accuracy_a"] == gene["probes"][0]["readings"]["accuracy_a"]
    )
    assert rows["random"]["final"] == {"accuracy_a": 0.25, "accuracy_b": 0.25}
    assert chamber.main(["--verify", str(path)]) == 0
    assert chamber.main(["--report", str(path)]) == 0
    out = capsys.readouterr().out
    assert "Ring `full`" in out and "`frozen`" in out
    assert chamber.main(["--report", str(path), "--curves", "full", "gene"]) == 0


def test_the_ring_pays_the_nursery_action_it_owes_and_counts_only_learned_steps(small_receipt):
    _, body = small_receipt
    nursery = body["nurseries"][0]
    assert nursery["totals"]["moments"] == 400 and nursery["owed"] in (0.0, 1.0)
    gene = next(row for row in body["rows"] if row["arm"] == "gene")
    assert gene["door"]["owed"] == nursery["owed"]
    assert gene["consistency"]["association->motor"]["count"] == gene["totals"]["learned"]
    assert "mean_dopamine" in gene["totals"] and "mean_delta" in gene["totals"]


def test_frozen_readings_use_one_temperature_for_every_arm(protocol, tmp_path):
    brain = chamber.make_brain(protocol["brain"], 0)
    world = chamber.make_world(0)
    temperature = float(protocol["brain"]["temperature"])
    base = chamber.frozen_readings(brain, world["patterns"], world["correct"], temperature)
    retuned = cd.Brain.load(brain.save(tmp_path / "b.npz"))
    chamber.enter_ring(retuned, "stage", protocol["stages"])
    staged = chamber.frozen_readings(retuned, world["patterns"], world["correct"], temperature)
    assert staged["reset"]["p_correct"] == pytest.approx(base["reset"]["p_correct"])
    assert staged["reset"]["margin"] == pytest.approx(base["reset"]["margin"])


def test_a_tampered_receipt_fails_verification(small_receipt, tmp_path):
    path, _ = small_receipt
    stored = json.loads(gzip.decompress(path.read_bytes()))
    stored["body"]["rows"][0]["final"]["accuracy_a"] = 1.0
    forged = tmp_path / "forged.json.gz"
    forged.write_bytes(gzip.compress((json.dumps(stored, sort_keys=True) + "\n").encode()))
    assert chamber.main(["--verify", str(forged)]) == 1


def test_learning_overrides_reach_the_brain_and_the_founder_has_none(protocol):
    brain = chamber.make_brain(protocol["brain"], 0)
    assert brain.describe()["learning"]["homeostasis_rate"] == 0.0
    point = {**protocol["brain"], "learning": {"homeostasis_rate": 0.02, "homeostasis_target": 0.3}}
    brain = chamber.make_brain(point, 0)
    learning = brain.describe()["learning"]
    assert (learning["homeostasis_rate"], learning["homeostasis_target"]) == (0.02, 0.3)


def test_the_arena_founder_brain_is_the_declared_one(protocol):
    brain = chamber.make_brain(protocol["brain"], 3)
    actor = brain.basal_ganglia.config
    assert (actor.eta, actor.eta_bias, actor.lam, actor.gamma, actor.eta_critic) == (
        0.03,
        0.003,
        0.6,
        0.95,
        5.0,
    )
    assert brain.learner.config.temperature == 0.3
    assert brain.arousal is not None and brain.arousal.config.youth == 300
    assert brain.working_memory is not None and brain.working_memory.amplitude == 0.3
    assert (
        isinstance(brain.hippocampus, cd.SynapticMemory) if hasattr(cd, "SynapticMemory") else True
    )
    assert len(brain.connectome.populations["module_0"]) == 48
    assert len(brain.connectome.populations["association"]) == 24
