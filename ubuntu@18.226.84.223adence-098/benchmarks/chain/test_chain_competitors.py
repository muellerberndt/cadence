"""The competitor arms of the prerequisite-chain nursery: the learners' eligibilities, guard
and determinism, what the arms declare, and the receipts that carry them."""

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
competitors = chamber.competitors
INPUTS, ACTIONS = chamber.INPUTS, 6
SIZES = {"elman": 647, "gru": 1703}  # 16 inputs, 6 actions, 16 hidden


@pytest.fixture(scope="module")
def protocol():
    protocol, _ = chamber.load_protocol(chamber.PROTOCOL)
    return protocol


def onehot(index: int, width: int = INPUTS) -> np.ndarray:
    x = np.zeros(width)
    x[index] = 1.0
    return x


@pytest.mark.parametrize("cell", competitors.CELLS)
def test_eligibilities_are_gradients_of_the_executed_policy_and_value(cell):
    life = competitors.ActorCritic(
        5, 3, 4, cell=cell, hidden=3, alpha=0.2, alpha_value=0.3, lam=0.0, clip=1e6
    )
    life.wv[:] = [0.1, -0.2, 0.3]
    life.Wa[:] = np.random.default_rng(1).normal(0.0, 0.5, (3, 3))
    life.act(onehot(1, 5), None)
    previous = life.h.copy()
    x = onehot(3, 5)
    action = life.act(x, 0.8)  # includes the nonterminal feedback update of this moment
    h, probabilities, value, _ = life.evaluate(x, previous)
    np.testing.assert_allclose(life.h, h)
    assert life.probs[action] == probabilities[action] and life.memo == value
    assert life.work["updates"] == 1 and life.last_delta is not None
    # Finite differences hold the previous hidden state fixed, as the declared one-moment
    # semi-gradient does; with lam 0 the trace is this moment's gradient alone.
    epsilon = 1e-6
    for trace, actor in ((life.actor_trace, True), (life.value_trace, False)):
        for name, recorded in trace.items():
            parameter = life.parameters()[name]
            for index in np.ndindex(recorded.shape):
                old = float(parameter[index])
                values = []
                for sign in (-1, 1):
                    parameter[index] = old + sign * epsilon
                    _, policy, estimate, _ = life.evaluate(x, previous)
                    values.append(np.log(policy[action]) if actor else estimate)
                parameter[index] = old
                assert recorded[index] == pytest.approx(
                    (values[1] - values[0]) / (2 * epsilon), abs=2e-9
                )


@pytest.mark.parametrize("cell", competitors.CELLS)
def test_the_guard_bounds_the_whole_actor_and_critic_gradient(cell):
    actor = competitors.ActorCritic(INPUTS, ACTIONS, 0, cell=cell, clip=1.0, lam=0.0)
    for weights in actor.cell.parameters().values():
        weights.fill(1.0)
    actor.Wa[:] = np.random.default_rng(2).normal(0.0, 1.0, actor.Wa.shape)
    actor.act(onehot(0), None)
    norm = np.sqrt(sum(np.sum(v * v) for v in actor.actor_trace.values()))
    assert norm == pytest.approx(1.0) and actor.work["clipped"] == 1
    critic = competitors.ActorCritic(INPUTS, ACTIONS, 0, cell=cell, clip=5.0, lam=0.0)
    critic.wv.fill(10.0)
    critic.act(onehot(2), None)
    norm = np.sqrt(sum(np.sum(v * v) for v in critic.value_trace.values()))
    assert norm == pytest.approx(5.0) and critic.work["value_clipped"] == 1
    # the TD error is bounded by the same guard
    critic.memo = 1000.0
    critic.act(onehot(2), 0.0)
    assert critic.last_delta == -5.0 and critic.work["delta_clipped"] == 1


@pytest.mark.parametrize("cell", competitors.CELLS)
def test_a_terminal_mark_clears_the_state_and_its_absence_carries_it(cell):
    life = competitors.ActorCritic(INPUTS, ACTIONS, 4, cell=cell, alpha=0.0, alpha_value=0.0)
    life.act(onehot(1), None)
    twin = life.clone(cleared=False)
    previous = life.h.copy()
    x = onehot(2)
    life.act(x, 0.0, done=True)
    twin.act(x, 0.0)
    np.testing.assert_allclose(life.h, life.evaluate(x, np.zeros_like(previous))[0])
    np.testing.assert_allclose(twin.h, twin.evaluate(x, previous)[0])
    assert not np.array_equal(life.h, twin.h)
    assert not any(v.any() for v in life.actor_trace.values()) or life.work["updates"] == 1


@pytest.mark.parametrize("cell", competitors.CELLS)
def test_seeded_lives_are_deterministic_and_seeds_differ(cell):
    def live(seed: int):
        learner = competitors.ActorCritic(INPUTS, ACTIONS, seed, cell=cell)
        rng = np.random.default_rng(9)
        reward = None
        actions = []
        for _ in range(200):
            actions.append(learner.act(onehot(int(rng.integers(INPUTS))), reward))
            reward = float(rng.random() < 0.3)
        return actions, learner

    first, one = live(3)
    second, two = live(3)
    third, _ = live(4)
    assert first == second and first != third
    for a, b in zip(one.parameters().values(), two.parameters().values(), strict=True):
        assert np.array_equal(a, b)
    assert one.work == two.work and one.work["updates"] == 199
    assert one.parameter_count == SIZES[cell]


def test_competitor_arms_change_only_what_they_declare(protocol, tmp_path):
    world = chamber.make_world(0, 4)
    elman = chamber.make_competitor("elman", protocol, 0, world["actions"])
    gru = chamber.make_competitor("gru", protocol, 0, world["actions"])
    assert elman.cell.name == "elman" and gru.cell.name == "gru"
    assert (elman.parameter_count, gru.parameter_count) == (SIZES["elman"], SIZES["gru"])
    declared = protocol["competitors"]
    for name in competitors.SETTINGS:
        assert getattr(elman, name) == getattr(gru, name) == declared["elman"][name]
        assert declared["gru"][name] == declared["elman"][name]
    assert np.array_equal(elman.Wa, gru.Wa) and not elman.Wa.any()  # readouts born alike
    # a competitor in the plan leaves the other arms' rows as they are
    short = {**protocol, "moments": 200, "probe_every": 100, "probe_moments": 50}
    alone = chamber.run(short, ["tabular", "random"], ["observed"], ["end"], [0], 1, tmp_path / "a")
    beside = chamber.run(
        short, ["tabular", "random", "elman"], ["observed"], ["end"], [0], 1, tmp_path / "b"
    )

    def rows(results):
        return {
            r["arm"]: {k: v for k, v in r.items() if k != "seconds"}
            for r in results["rows"]
            if r["kind"] == "control"
        }

    assert rows(alone) == rows(beside)
    assert [r["arm"] for r in beside["rows"] if r["kind"] == "competitor"] == ["elman"]


def test_probes_read_cleared_copies_and_leave_the_life_untouched(protocol):
    world = chamber.make_world(1, 4)
    learner = chamber.make_competitor("gru", protocol, 1, world["actions"])
    reward = None
    rng = np.random.default_rng(5)
    stage = 0
    for _ in range(50):
        action = learner.act(chamber.observe(world, stage, "hidden"), reward)
        reward, stage, _, _ = chamber.outcome(world, stage, action, "end", rng)
    state, work = learner.h.copy(), dict(learner.work)
    readings, probe_work = chamber.competitor_readings(learner, world, "hidden", 100, 1)
    assert np.array_equal(learner.h, state) and learner.work == work and state.any()
    assert probe_work["probe_forward_passes"] == 2 * 100 + 4
    assert set(readings) == {"greedy", "sampled", "stages", "greedy_stage_accuracy", "p_right_mean"}
    assert len(readings["stages"]) == 4
    # the hidden mode shows every stage the same observation: one answer from a cleared state
    assert len({s["greedy"] for s in readings["stages"]}) == 1


def test_a_short_run_writes_a_verifiable_receipt_with_the_competitor_arms(tmp_path):
    protocol, _ = chamber.load_protocol(chamber.PROTOCOL)
    protocol["moments"], protocol["probe_every"], protocol["probe_moments"] = 400, 200, 100
    arms = ["elman", "gru", "tabular"]
    results = chamber.run(
        protocol, arms, ["observed", "hidden"], ["end"], [0], 1, tmp_path / "probes"
    )
    assert len(results["rows"]) == 6
    for row in results["rows"]:
        if row["kind"] != "competitor":
            continue
        assert len(row["probes"]) == 3 and row["totals"]["moments"] == 400
        assert row["totals"]["learned"] == row["work"]["updates"] == 399
        assert row["work"]["parameters"] == row["describe"]["parameters"] == SIZES[row["arm"]]
        assert row["work"]["hidden"] == 16 and row["work"]["multiply_accumulates"] > 0
        assert set(row["final"]) >= {"greedy", "sampled", "stages", "greedy_stage_accuracy"}
    body = {"schema": chamber.SCHEMA, "protocol": protocol, "seeds": [0], "arms": arms, **results}
    out = tmp_path / "receipt.json.gz"
    manifest = chamber.Receipt.build(chamber.SCHEMA, {}, chamber.sources()).source
    chamber.write_receipt(out, body, manifest)
    valid, reason = chamber.verify(out, current=True)
    assert valid, reason
    stored = json.loads(gzip.decompress(out.read_bytes()))
    files = [item["path"] for item in stored["source"]["files"]]
    assert files[:2] == ["prerequisite_chain.py", "competitors.py"]
    report = chamber.markdown(chamber.read_receipt(out)["body"])
    assert "| observed | end | elman |" in report and "| hidden | end | gru |" in report
    stored["body"]["rows"][0]["work"]["updates"] = 0
    bad = tmp_path / "tampered.json.gz"
    bad.write_bytes(gzip.compress(json.dumps(stored).encode()))
    assert not chamber.verify(bad)[0]


def test_the_command_line_refuses_an_undeclared_competitor(tmp_path, capsys):
    protocol, _ = chamber.load_protocol(chamber.PROTOCOL)
    del protocol["competitors"]
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol))
    with pytest.raises(SystemExit):
        chamber.main(["--protocol", str(path), "--arms", "elman", "--seeds", "0", "--workers", "1"])
    capsys.readouterr()
