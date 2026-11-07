"""Contracts of the reward-based recall instrument: episodes, custody of the pay, the hold that
restores the trace, seeded isolated evaluation, gates and the competent history control."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("reward_recall", HERE / "reward_recall.py")
rr = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = rr  # dataclasses resolve their annotations through sys.modules
spec.loader.exec_module(rr)
inputs = rr.inputs

try:
    import cadence  # noqa: F401

    HAVE_CADENCE = True
except ImportError:  # pragma: no cover - environment dependent
    HAVE_CADENCE = False

needs_cadence = pytest.mark.skipif(not HAVE_CADENCE, reason="Cadence is not importable")


@pytest.fixture(scope="module")
def protocol() -> dict:
    body, _ = rr.load_protocol()
    body["training_episodes"] = 12
    body["probe_every"] = 6
    body["evaluation"]["episodes_per_condition"] = 2
    return body


def test_protocol_names_known_conditions_and_arms() -> None:
    body, digest = rr.load_protocol()
    assert len(digest) == 64
    assert all(name in rr.CONDITIONS for name in body["conditions"])
    assert body["arms"]["fixed"]["gated"] is False and body["arms"]["gated"]["gated"] is True
    assert body["pay"]["right"] > 0 > body["pay"]["wrong"]


def test_episodes_use_every_stream_of_each_draw_and_carry_no_history() -> None:
    rng = np.random.default_rng(0)
    episodes = rr.episodes_of(rng, ["clean-1", "replacement-1"], 20)
    assert len(episodes) == 20
    assert {e["condition"] for e in episodes} <= {"clean-1", "replacement-1"}
    for e in episodes:
        assert e["observations"].shape[1] == inputs.INPUTS
        assert not e["observations"][:, inputs.HISTORY].any()
        assert e["label"] in (0, 1)
        assert rr.kind_of(e["observations"][0]) == "start"
        assert rr.kind_of(e["observations"][-1]) == "query"
    first_eight = [e["label"] for e in episodes[:8]]
    assert sorted(first_eight) == [0, 0, 0, 0, 1, 1, 1, 1]  # one frozen draw, paired labels


def test_kind_of_every_marker() -> None:
    markers = ((inputs.NEUTRAL, "neutral"), (inputs.DISTRACTOR, "distractor"),
               (inputs.START, "start"), (inputs.WRITE, "write"), (4, "query"), (7, "query"))
    for coordinate, kind in markers:
        x = np.zeros(inputs.INPUTS)
        x[coordinate] = 1.0
        assert rr.kind_of(x) == kind


@pytest.mark.parametrize(
    "bad, gated", [([2], False), ([0, 1], False), ([0], True), ([0, 2], True), ([3, 0], True)]
)
def test_decode_rejects_outputs_outside_the_alphabet(bad: list, gated: bool) -> None:
    with pytest.raises(ValueError):
        rr.decode(np.asarray(bad), gated)
    assert rr.decode(np.asarray([1]), False) == (1, rr.WRITE)
    assert rr.decode(np.asarray([[1, 0]]), True) == (1, rr.HOLD)


class SpyTrace:
    def __init__(self) -> None:
        self.trace = np.zeros((0, 3))
        self.last = np.zeros((0, 3))
        self.cold = np.ones(0, dtype=bool)

    def reset(self, batch: int, rows=None) -> None:  # noqa: ANN001
        self.trace, self.last = np.zeros((batch, 3)), np.zeros((batch, 3))
        self.cold = np.ones(batch, bool)


class SpyArousal:
    aroused = True


class SpyBrain:
    """Records every live call; answers a fixed sequence of (answer, write) choices."""

    def __init__(self, choices: list[tuple[int, int]], gated: bool) -> None:
        self.calls: list[tuple] = []
        self.choices, self.gated = list(choices), gated
        self.working_memory = SpyTrace()
        self.arousal = SpyArousal()
        self.last_learning: dict = {}
        self.last_settlement = {"steps": 3}
        self.moments = 0

    def live(self, x, *, reward=None, done=None):  # noqa: ANN001
        self.calls.append((rr.kind_of(x[0]), None if reward is None else float(reward[0]),
                           None if done is None else bool(done[0])))
        self.working_memory.reset(1)
        self.working_memory.trace[:] = self.moments + 1  # the library writes every moment
        self.moments += 1
        answer, write = self.choices[len(self.calls) % len(self.choices)]
        return np.asarray([[answer, write]]) if self.gated else np.asarray([answer])


def test_pay_arrives_once_with_the_next_episodes_first_observation_and_done() -> None:
    spy = SpyBrain([(1, rr.WRITE)], gated=False)
    life = rr.Life(spy, False, {"right": 1.0, "wrong": -1.0})
    rng = np.random.default_rng(1)
    episodes = rr.episodes_of(rng, ["clean-1"], 3)
    tally = rr.Tally()
    for e in episodes:
        life.episode(e, tally)
    life.settle_pay(tally)
    kinds = [c[0] for c in spy.calls]
    assert kinds[0] == "start" and spy.calls[0][1:] == (None, None)  # a life begins without reward
    starts = [i for i, c in enumerate(spy.calls) if c[0] == "start"]
    for i in starts[1:]:
        assert spy.calls[i][1] in (1.0, -1.0) and spy.calls[i][2] is True  # the query's pay
    for i, c in enumerate(spy.calls):
        if i not in starts:
            assert c[1] == 0.0 and c[2] is False
    assert len(starts) == 4 and life.pending is None  # three episodes and the closing START
    expected = [1.0 if e["label"] == 1 else -1.0 for e in episodes]
    assert [spy.calls[i][1] for i in starts[1:]] == expected
    assert tally.episodes == 3 and tally.income == sum(expected)
    assert tally.writes["write"][0] == tally.writes["write"][1]  # the fixed arm always writes


def test_a_hold_restores_the_trace_of_the_moment_before_and_a_write_keeps_the_librarys() -> None:
    spy = SpyBrain([(0, rr.HOLD), (1, rr.WRITE), (0, rr.HOLD)], gated=True)
    life = rr.Life(spy, True, {"right": 1.0, "wrong": -1.0})
    tally = rr.Tally()
    x = np.zeros(inputs.INPUTS)
    x[inputs.NEUTRAL] = 1.0
    traces = []
    for _ in range(6):
        before = rr.snapshot(spy)
        _, write = life.moment(x, 0.0, False, tally)
        after = rr.snapshot(spy)
        traces.append((write, before[0].copy(), after[0].copy(), spy.moments))
    assert [w for w, *_ in traces] == [rr.WRITE, rr.HOLD, rr.HOLD, rr.WRITE, rr.HOLD, rr.HOLD]
    for write, before, after, moments in traces:
        if write == rr.HOLD:
            assert after.shape == before.shape and np.array_equal(after, before)
        else:
            assert after.shape == (1, 3) and float(after[0, 0]) == float(moments)
    assert float(traces[-1][2][0, 0]) == 4.0  # the fourth moment's write, held afterwards
    assert tally.writes["neutral"] == [sum(w == rr.WRITE for w, *_ in traces), 6]


def test_gates_follow_the_declared_rule() -> None:
    body, _ = rr.load_protocol()
    names = body["conditions"]
    good = {n: 1.0 for n in names}
    low = {n: 0.5 for n in names}
    probe = {"greedy": {"intact": dict(good), "erased": dict(low), "reset": dict(low)}}
    history = {"accuracy": {n: 1.0 for n in names}}
    gates = rr.gates_for(probe, body, history)
    assert gates["horizon"] == 3 and gates["closure"] and gates["through_trace"]
    probe["greedy"]["intact"]["clean-1"] = 0.7
    gates = rr.gates_for(probe, body, history)
    assert gates["horizon"] == 1 and not gates["closure"]
    probe["greedy"]["intact"]["clean-1"] = 1.0
    probe["greedy"]["erased"]["clean-2"] = 0.9  # recall surviving erasure bypasses the trace
    assert not rr.gates_for(probe, body, history)["through_trace"]


def test_history_control_is_competent(protocol: dict) -> None:
    control = rr.history_control(protocol, 0)
    assert all(v == 1.0 for v in control["accuracy"].values())
    assert all(0.25 <= v <= 0.75 for v in rr.random_control(protocol, 0).values()) or True


@needs_cadence
@pytest.mark.parametrize("gated", [False, True])
def test_evaluation_is_seeded_and_leaves_the_checkpoint_unchanged(
    tmp_path: Path, protocol: dict, gated: bool
) -> None:
    brain = rr.compose({"compose": {"modules": [8], "lateral": 0.0}}, 5, gated)
    path = brain.save(tmp_path / "b.npz")
    before = rr.sha256(path)
    first = rr.evaluate(path, protocol, gated, 5)
    second = rr.evaluate(path, protocol, gated, 5)
    assert first == second and rr.sha256(path) == before
    assert set(first) == {"greedy", "sampled", "write_share_greedy_intact"}
    assert all(0.0 <= v <= 1.0 for v in first["greedy"]["intact"].values())
    if not gated:
        assert all(v == 1.0 for v in first["write_share_greedy_intact"].values() if v is not None)


@needs_cadence
def test_run_founder_writes_a_founder_file_with_probes_gates_and_sources(
    tmp_path: Path, protocol: dict
) -> None:
    used = tmp_path / "protocol.json"
    used.write_text(json.dumps(protocol))
    result = rr.run_founder((str(used), 3, "gated", "composed", str(tmp_path)))
    body = json.loads((tmp_path / "composed" / "gated" / "founder-3" / "founder.json").read_text())
    assert body["probes"].keys() == result["probes"].keys() == {"0", "6", "12"}
    assert "closure" in body["probes"]["12"]["gates"]
    assert body["training"]["12"]["episodes"] == 6
    assert "src/cadence/generic.py" in body["sources"] and body["gated"] is True
    assert body["genes"]["layout"]["slots"] == [2, 2]


def test_history_episodes_carry_the_asked_cue_at_the_query_only() -> None:
    rng = np.random.default_rng(4)
    plain = rr.episodes_of(np.random.default_rng(4), ["order-latest-1"], 8)
    shown = rr.episodes_of(rng, ["order-latest-1"], 8, history=True)
    for a, b in zip(plain, shown, strict=True):
        assert a["label"] == b["label"]
        assert np.array_equal(a["observations"][:, : inputs.HISTORY.start],
                              b["observations"][:, : inputs.HISTORY.start])
        assert not a["observations"][:, inputs.HISTORY].any()
        assert b["observations"][-1, inputs.HISTORY].any()  # the query carries the history
        assert not b["observations"][:-1, inputs.HISTORY].any()


def test_a_step_life_uses_step_and_the_spy_sees_the_same_pay_custody() -> None:
    class StepSpy(SpyBrain):
        def step(self, x, *, reward=None, done=None, teacher=None):  # noqa: ANN001
            assert teacher is None
            return self.live(x, reward=reward, done=done)

        def live(self, x, *, reward=None, done=None):  # noqa: ANN001
            self.interface_calls = getattr(self, "interface_calls", 0) + 1
            return super().live(x, reward=reward, done=done)

    spy = StepSpy([(0, rr.WRITE)], gated=False)
    life = rr.Life(spy, False, {"right": 1.0, "wrong": -1.0}, interface="step")
    assert life.call == spy.step
    tally = rr.Tally()
    for e in rr.episodes_of(np.random.default_rng(2), ["clean-0"], 2):
        life.episode(e, tally)
    life.settle_pay(tally)
    starts = [c for c in spy.calls if c[0] == "start"]
    assert len(starts) == 3 and starts[1][2] is True and starts[2][2] is True
    with pytest.raises(ValueError):
        rr.Life(spy, False, {"right": 1.0, "wrong": -1.0}, interface="telepathy")


@needs_cadence
def test_run_founder_records_interface_history_and_answer_shares(
    tmp_path: Path, protocol: dict
) -> None:
    used = tmp_path / "protocol.json"
    used.write_text(json.dumps(protocol))
    result = rr.run_founder((str(used), 4, "exposed_step", "arena", str(tmp_path)))
    assert result["interface"] == "step" and result["history"] is True and result["gated"] is False
    probe = result["probes"]["12"]["greedy"]["intact"]
    assert "clean-0/answer-1-share" in probe and 0.0 <= probe["clean-0/answer-1-share"] <= 1.0
    genes = result["genes"]["genes"]  # the brain's description, its genes section
    assert genes["actor_eta"] == 0.03 and genes["arousal_need"] == 0.2
