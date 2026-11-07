"""The two-cue chamber (cadence#208, cadence#209): the task, its panels, the two information
conditions as the brain receives them, readings that never change the life, and the
preservation chambers' setting wrapper."""

from __future__ import annotations

import importlib.util
import itertools
import json
import warnings
from pathlib import Path

import numpy as np
import pytest

import cadence as cd


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tc = _load("two_cue")
pr = _load("preservation")


def test_the_cues_take_the_two_actions_counterbalanced_by_the_founders_bits():
    assert [tc.label_of(c, 0) for c in range(4)] == [0, 1, 0, 1]
    assert [tc.label_of(c, 1) for c in range(4)] == [1, 0, 0, 1]
    assert [tc.label_of(c, 2) for c in range(4)] == [0, 1, 1, 0]
    assert [tc.label_of(c, 3) for c in range(4)] == [1, 0, 1, 0]


def test_an_episode_is_the_cue_the_distractors_and_the_identical_suffix():
    first, second = tc.events_of(0, (2, 5)), tc.events_of(1, (2, 5))
    assert first.shape == (5, tc.INPUTS) and (first.sum(axis=1) == 1).all()
    assert [int(np.argmax(r)) for r in first] == [0, 6, 9, tc.S1, tc.S2]
    np.testing.assert_array_equal(first[1:], second[1:])  # only the cue tells them apart
    assert tc.events_of(1, ()).shape == (3, tc.INPUTS)


def test_one_tuple_in_four_is_held_out_from_delay_two():
    assert not any(tc.held_out(0, t) for t in itertools.product(range(6), repeat=1))
    pairs = list(itertools.product(range(6), repeat=2))
    for founder in range(6):
        held = sum(tc.held_out(founder, t) for t in pairs)
        assert 4 <= held <= 16
    quads = list(itertools.product(range(6), repeat=4))
    assert abs(np.mean([tc.held_out(0, t) for t in quads]) - 0.25) < 0.03


def test_the_schedule_pairs_the_cues_and_never_trains_a_held_out_tuple():
    plan = tc.schedule(3, 2, 512)
    cues = np.array([c for c, _ in plan])
    assert sorted(set(cues)) == [0, 1]
    assert (cues.reshape(-1, 2).sum(axis=1) == 1).all()  # both cues in every block of two
    assert not any(tc.held_out(3, t) for _, t in plan)
    assert plan == tc.schedule(3, 2, 512) and plan != tc.schedule(4, 2, 512)
    assert all(t == () for _, t in tc.schedule(0, 0, 8))


@pytest.mark.parametrize("delay", [0, 1, 2, 4, 8])
def test_the_panels(delay):
    train, nothing = tc.panel(5, delay, "train")
    held, _ = tc.panel(5, delay, "held_out")
    assert nothing == (delay < 2)
    for episodes in (train, held):
        cues = [c for c, _ in episodes]
        assert cues == [0, 1] * (len(episodes) // 2)  # every tuple with both cues
    tuples_train = {t for _, t in train}
    tuples_held = {t for _, t in held}
    if delay < 2:
        assert train == held and len(tuples_train) == 6**delay
    else:
        assert not tuples_train & tuples_held
        assert all(tc.held_out(5, t) for t in tuples_held)
        assert not any(tc.held_out(5, t) for t in tuples_train)
    if delay == 2:
        assert len(tuples_train) + len(tuples_held) == 36
    if delay >= 4:
        assert len(tuples_train) == len(tuples_held) == tc.PANEL


class Recorder:
    """A stand-in brain that records what the conditions give it."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def __deepcopy__(self, memo):  # a fork records apart from the life
        return Recorder()

    def step(self, x, reward=None, done=None, teacher=None):
        self.calls.append(("step", int(np.argmax(x[0])), reward, done, teacher))
        return np.array([0])

    def act(self, x, greedy=False):
        self.calls.append(("act", int(np.argmax(x[0])), greedy))
        return np.array([0])

    def reset(self):
        self.calls.append(("reset",))

    def describe(self):
        return {"genes": {}}

    @property
    def memories(self):
        return {}

    @property
    def basal_ganglia(self):
        class Policy:
            state = None

            @staticmethod
            def probabilities(state, bias=None):
                return np.array([[0.75, 0.25]])

        return Policy()


def test_the_reward_condition_pays_the_answer_with_the_next_episodes_first_event(monkeypatch):
    brain = Recorder()
    monkeypatch.setattr(tc, "make_brain", lambda arm, founder, settings=None: brain)
    monkeypatch.setattr(tc, "read_panel", lambda *a, **k: {"accuracy": 0.5})
    row = tc.run_life({"condition": "reward", "arm": "founder", "founder": 1, "delay": 1,
                       "budgets": [3]})
    assert "error" not in row and row["work"]["steps"] == 12
    plan = tc.schedule(1, 1, 3)
    steps = [c for c in brain.calls if c[0] == "step"]
    assert steps[0][2:] == (None, None, None)  # the first moment has no outcome
    for episode, (cue, _) in enumerate(plan):
        first, *rest = steps[4 * episode : 4 * episode + 4]
        assert first[1] == cue and [c[1] for c in rest][1:] == [tc.S1, tc.S2]
        if episode:
            earlier = plan[episode - 1][0]
            owed = 1.0 if tc.label_of(earlier, 1) == 0 else -1.0  # the recorder answers 0
            assert first[2] == [owed] and first[3] == [True]
        assert all(c[2] == [0.0] and c[3] is None and c[4] is None for c in rest)
    assert not any(c[0] in ("act", "reset") for c in brain.calls)


def test_the_lesson_condition_teaches_the_answer_alone_on_a_fresh_stream(monkeypatch):
    brain = Recorder()
    monkeypatch.setattr(tc, "make_brain", lambda arm, founder, settings=None: brain)
    monkeypatch.setattr(tc, "read_panel", lambda *a, **k: {"accuracy": 0.5})
    row = tc.run_life({"condition": "lesson", "arm": "founder", "founder": 2, "delay": 2,
                       "budgets": [2]})
    assert "error" not in row and row["work"]["lessons"] == 2
    assert len(brain.calls) == 12
    for (cue, tuple_), start in zip(tc.schedule(2, 2, 2), (0, 6), strict=True):
        calls = brain.calls[start : start + 6]
        assert calls[0] == ("reset",)
        assert [c[1] for c in calls[1:5]] == [cue, *(tc.DISTRACTOR + d for d in tuple_), tc.S1]
        assert all(c[0] == "act" and c[2] for c in calls[1:5])  # greedy acts
        assert calls[5][:2] == ("step", tc.S2) and calls[5][4] == [tc.label_of(cue, 2)]
        assert calls[5][2] is None and calls[5][3] is None


@pytest.mark.parametrize(
    "condition,arm,delay", [("reward", "slow_episodic", 0), ("lesson", "recall_recency0", 2)]
)
def test_a_reading_never_changes_the_life(condition, arm, delay):
    base = {"condition": condition, "arm": arm, "founder": 0, "delay": delay}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        read_twice = tc.run_life({**base, "budgets": [6, 12]})
        read_once = tc.run_life({**base, "budgets": [12]})
    assert "error" not in read_twice and "error" not in read_once
    assert json.dumps(read_twice["readings"][-1]["held_out"]) == json.dumps(
        read_once["readings"][-1]["held_out"]
    )
    assert read_twice["log"] == read_once["log"]
    reading = read_once["readings"][-1]["held_out"]
    assert len(reading["separation"]) == delay + 3 and 0.0 <= reading["accuracy"] <= 1.0
    assert (reading["vote_one"] is not None) == (arm == "slow_episodic")


def _lived(brain, cue, tuple_=(1, 2)):
    """A fresh stream of the brain up to the answer of one episode, every act greedy."""
    brain.reset()
    for x in tc.events_of(cue, tuple_)[:-1]:
        brain.act(x[None, :], greedy=True)


def test_an_erasure_empties_every_carrier_it_joins():
    assert tc.carriers("working_memory+recall") == ("working_memory", "recall")
    brain = tc.make_brain("recall_recency0", 0)
    _lived(brain, 0)
    trace, recall = brain.memories["working_memory"], brain.memories["recall"]
    assert trace.trace.any() and recall.slots == 4 and brain.basal_ganglia.state is not None
    tc.empty(brain, "working_memory+recall")
    assert not trace.trace.any() and recall.slots == 0
    assert brain.basal_ganglia.state is not None  # the settled activity is a carrier of its own
    tc.empty(brain, "activity")
    assert brain.basal_ganglia.state is None
    assert tc.carries(brain, "working_memory+recall+activity")
    assert not tc.carries(tc.make_brain("founder", 0), "working_memory+recall")


def test_a_transplant_of_every_carrier_settles_the_answer_as_the_donor():
    brain = tc.make_brain("recall_recency0", 0)
    _lived(brain, 0)
    donor = tc.carried(brain, "working_memory+recall+activity")
    answer = brain.act(tc.events_of(0, (1, 2))[-1][None, :], greedy=True)
    settled = np.array(brain.basal_ganglia.state.activation)
    _lived(brain, 1)
    own = np.array(brain.memories["working_memory"].trace)
    tc.carry(brain, {"working_memory": donor["working_memory"]})
    given = donor["working_memory"]["trace"]
    np.testing.assert_array_equal(brain.memories["working_memory"].trace, given)
    assert not np.array_equal(own, given)
    tc.carry(brain, donor)
    np.testing.assert_array_equal(
        brain.act(tc.events_of(1, (1, 2))[-1][None, :], greedy=True), answer
    )
    np.testing.assert_array_equal(brain.basal_ganglia.state.activation, settled)


def test_the_erasures_and_transplants_are_read_at_the_last_budget_and_change_nothing():
    base = {"condition": "lesson", "arm": "recall_recency0", "founder": 0, "delay": 2,
            "budgets": [4, 8]}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plain = tc.run_life(base)
        read = tc.run_life({**base, "erase": ["working_memory+recall", "history"],
                            "transplant": ["recall", "working_memory+recall+activity"]})
    assert "error" not in read
    first, last = read["readings"]
    assert 0.0 <= last["working_memory+recall_erased"] <= 1.0 and "history_erased" not in last
    moved = last["working_memory+recall+activity_transplant"]
    assert moved["episodes"] == last["held_out"]["episodes"]
    # within a pair the cues take the two actions: an answer follows the donor or the recipient
    assert moved["follows_donor"] + moved["accuracy"] == pytest.approx(1.0)
    assert "recall_transplant" in last and not any(k.endswith(("_erased", "_transplant"))
                                                   for k in first)
    assert json.dumps(plain["readings"][-1]["held_out"]) == json.dumps(last["held_out"])
    assert plain["log"] == read["log"]


def test_every_arm_composes_and_the_hold_and_extinction_reach_the_episodic_memory():
    for arm in tc.ARMS:
        brain = tc.make_brain(arm, 0)
        assert brain.basal_ganglia.config.eta == tc.ARMS[arm].get("actor_eta", 1.0)
    assert tc.make_brain("slow_episodic_hold", 0).memories["episodic"].hold == "unchanged"
    assert tc.make_brain("slow_episodic_extinction", 0).memories["episodic"].extinction == 0.5
    assert tc.make_brain("recall_recency0_boundary", 0).describe()["genes"]["recall_boundary"]


def test_the_summary_reads_the_gate():
    rows = [
        {"condition": "reward", "arm": "a", "delay": 0, "founder": f, "budgets": [4],
         "readings": [{"held_out": {"accuracy": 1.0 if f else 0.5}}],
         "verdict": {"held_out": 1.0 if f else 0.5}}
        for f in range(6)
    ]
    (cell,) = tc.summarize(rows)
    assert cell["at_gate"] == [5] and cell["passes"] and cell["held_out_mean"] == [0.917]


def test_a_setting_reaches_each_chambers_own_brains_and_leaves_compose_as_it_was():
    nursery = pr._module("odour_nursery", pr.ROOT / "benchmarks/reversal/odour_nursery.py")
    point = json.loads(nursery.PROTOCOL.read_text())["operating_point"]
    with pr.composed_with({"memory_choice": True, "memory_extinction": 0.5}):
        inside = nursery.make_brain(point, 0, None)
    outside = nursery.make_brain(point, 0, None)
    assert inside.memory_choice and inside.memories["episodic"].extinction == 0.5
    assert not outside.memory_choice
    assert inside.basal_ganglia.config.eta == outside.basal_ganglia.config.eta == 0.1
    assert type(cd.Brain.__dict__["compose"]) is classmethod
    with pytest.raises(ValueError, match="sets 'seed' itself"):
        with pr.composed_with({"seed": 3}):
            cd.Brain.compose(1, 2, seed=0, recall=False)


def test_the_bandit_moves_its_paying_arm_every_block(monkeypatch):
    monkeypatch.setitem(pr.BANDIT, "trials", 40)
    monkeypatch.setitem(pr.BANDIT, "block", 20)
    monkeypatch.setitem(pr.BANDIT, "last", 10)
    monkeypatch.setitem(pr.BANDIT, "hold", 5)
    row = pr.run_life({"chamber": "bandit", "seed": 0, "settings": {}})
    assert "error" not in row and len(row["blocks"]) == 2
    assert all(0.0 <= b["share"] <= 1.0 for b in row["blocks"])


def _phase(**values):
    return {"final": 1.0, "lag": 10, "stable": 1.0, "aroused_late": 0.0, "fed": 1.0,
            "wrong": 0.0, **values}


def test_the_nurserys_checks_and_the_learning_checks_of_its_step_arm(monkeypatch):
    nursery = pr._module("odour_nursery", pr.ROOT / "benchmarks/reversal/odour_nursery.py")
    phases = [_phase(), _phase(final=0.85, lag=None), _phase(aroused_late=0.5)]
    monkeypatch.setattr(nursery, "run_life", lambda *a, **k: {"phases": phases})
    live = pr.nursery_life(0, {}, "live", 300)
    assert live["checks"] == {"acquired": True, "reversed": False, "returned": True,
                              "stable_kept": True, "calm": False}
    assert not live["passed"] and live["lag"] == [10, None, 10]
    phases[1] = _phase()
    assert not pr.nursery_life(0, {}, "live", 300)["passed"]  # not calm
    assert pr.nursery_life(0, {}, "step", 300)["passed"]  # calm is not a learning check


def test_the_key_doors_acquisition_and_its_joint_predicates(monkeypatch):
    keydoor = pr._module("key_door", pr.ROOT / "benchmarks/keydoor/key_door.py")
    phases = [_phase(fed=0.92), _phase(fed=0.5), _phase(lag=80)]
    monkeypatch.setattr(keydoor, "run_life", lambda *a, **k: {"phases": phases})
    row = pr.keydoor_life(0, {}, 2)
    assert row["passed"] and not row["joint"]
    assert row["checks"] == {"acquired": True, "adapted": False, "retained": False,
                             "frugal": True, "calm": True}
