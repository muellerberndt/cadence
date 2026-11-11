"""Guards of the key-door nursery: its protocol, its world, its arms, its custody and its gates."""

import gzip
import importlib.util
import json
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import cadence as cd

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("key_door", HERE / "key_door.py")
keydoor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(keydoor)


LEGACY_ARMS = ("live", "step", "lambda-zero", "yoked", "frozen", "blind", "tabular", "random")


@pytest.fixture(scope="module")
def protocol():
    return json.loads((HERE / "protocol-3.json").read_text())


@pytest.fixture(scope="module")
def protocol_1():
    return json.loads((HERE / "protocol.json").read_text())


@pytest.fixture(scope="module")
def quick(protocol):
    """The protocol with few trips per rule, for quick lives."""
    return {**protocol, "episodes": 60, "probe_every": 20}


def test_the_protocol_is_the_frozen_one(protocol, protocol_1):
    assert protocol_1["schema"] == keydoor.LEGACY_SCHEMA == "key-door/1"
    assert protocol["schema"] == keydoor.SCHEMA == "key-door/3"
    assert keydoor.LEGACY_SCHEMAS == ("key-door/1", "key-door/2")
    assert protocol["rules"] == ["chest", "lamp", "chest"]
    assert protocol["delays"] == [2, 5, 10] and protocol["episodes"] == 500
    assert (protocol["length"], protocol["jitter"]) == (15, 2)
    assert (protocol_1["length"], protocol_1["jitter"]) == (14, 1)
    assert (protocol["cost"], protocol["food"], protocol["truncation"]) == (0.25, 1.0, 0.05)
    seeds = protocol["seeds"]
    assert seeds["development"] == list(range(8))
    assert seeds["spent"] == list(range(700, 710)) + list(range(800, 810))  # both freezes
    assert seeds["confirmation"] == list(range(900, 910))
    sets = [set(seeds[name]) for name in ("development", "spent", "confirmation")]
    assert all(not a & b for i, a in enumerate(sets) for b in sets[i + 1 :])
    assert protocol["tabular"] == {"alpha": 0.5, "epsilon": 0.1, "gamma": 0.9, "lam": 0.8}
    assert {"hidden", "alpha", "alpha_value", "gamma", "lam", "temperature"} <= set(
        protocol["recurrent"]
    )
    founders = cd.ArousalConfig()
    genes = keydoor.arousal_genes(protocol["arousal"])
    assert cd.ArousalConfig(**genes) == replace(founders, need=0.03)
    point = {k: v for k, v in protocol["operating_point"].items() if k != "note"}
    assert point == protocol_1["operating_point"]  # the simplest existing System 1 stays
    copy = {k: v for k, v in protocol["copy"].items() if k != "note"}
    assert (
        set(copy) == {"efference_amplitude", "efference_decay"} and copy["efference_decay"] == 0.0
    )
    gates = protocol["gates"]
    assert (gates["fed"], gates["wrong"], gates["aroused_late"], gates["share"]) == (
        0.9,
        1.5,
        0.35,
        0.9,
    )
    assert gates["delays"] == [2, 5] and gates["retained_lag"] == 50


def test_every_trip_has_the_same_length_and_the_declared_order(protocol):
    rng = np.random.default_rng(0)
    for delay in protocol["delays"]:
        for _ in range(20):
            cells = keydoor.corridor(delay, protocol["length"], protocol["jitter"], rng)
            assert len(cells) == protocol["length"]
            levers = cells.count(keydoor.LEVER)
            assert abs(levers - delay) <= protocol["jitter"]
            floors = cells.count(keydoor.FLOOR)
            assert cells == [keydoor.FLOOR] * floors + [keydoor.CHEST, keydoor.LAMP] + [
                keydoor.LEVER
            ] * levers + [keydoor.DOOR]
    x = keydoor.observe(keydoor.LAMP, True, True)
    assert x.shape == (1, 6) and x[0, keydoor.LAMP] == 1.0 and x[0, 5] == 1.0
    assert keydoor.observe(keydoor.LAMP, True, False).shape == (1, 5)


def test_the_world_pays_the_door_with_the_key_and_a_cut_trip_ends_with_done_clear(
    quick, monkeypatch
):
    """The tabular learner's life is deterministic and cheap: read the world back through
    what it was handed. ``done`` is set on the observation after a door and on no other,
    so a cut trip carries its forecast over into the next (a truncated bootstrap)."""
    made = keydoor.make_life
    handed = []

    def recording(arm, protocol, seed, genes):
        life = made(arm, protocol, seed, genes)
        act = life.act

        def logged(kind, holding, reward, done):
            handed.append((kind, reward, done))
            return act(kind, holding, reward, done)

        life.act = logged
        return life

    monkeypatch.setattr(keydoor, "make_life", recording)
    row = keydoor.run_life("tabular", 0, 2, {**quick, "episodes": 40, "truncation": 0.5})
    for phase in row["phases"]:
        assert 0.0 <= phase["fed"] <= phase["took"] <= 1.0  # food needs the key
        assert phase["cut"] > 0 and phase["episodes"] + phase["cut"] == 40
        assert sum(map(sum, phase["visits"])) == phase["moments"] < 40 * quick["length"]
        assert phase["wrong"] >= 0.0
    assert [p["keyed"] for p in row["phases"]] == ["chest", "lamp", "chest"]
    assert handed[0][1:] == (None, False) and all(r is not None for _, r, _ in handed[1:])
    for (kind, _, _), (_, _, done) in zip(handed, handed[1:], strict=False):
        assert done == (kind == keydoor.DOOR)
    assert sum(k == keydoor.DOOR for k, _, _ in handed) == 3 * 40 - sum(
        p["cut"] for p in row["phases"]
    )


def test_the_operating_point_reaches_the_brain(protocol):
    """Every setting the protocol declares is on the living brain; the actor's bias rate
    follows the composed rule unless the point names it."""
    point = protocol["operating_point"]
    life = keydoor.make_life("live", protocol, 0, protocol["arousal"])
    actor, learner = life.brain.basal_ganglia.config, life.brain.learner.config
    for name in type(actor).__slots__:
        if name in point:
            assert getattr(actor, name) == point[name], name
    if "eta" in point and "eta_bias" not in point:
        assert actor.eta_bias == point["eta"] / 10
    for name, value in point.get("learner", {}).items():
        assert getattr(learner, name) == value, name
    founders = cd.ArousalConfig(**keydoor.arousal_genes(protocol["arousal"]))
    assert life.brain.arousal.config == founders
    genes = protocol["arousal"]
    assert keydoor.make_life("blind", protocol, 0, genes).pouch is False
    assert keydoor.make_life("lambda-zero", protocol, 0, genes).brain.basal_ganglia.config.lam == 0
    assert keydoor.make_life("step", protocol, 0, genes).brain.arousal is None
    assert life.brain.efference is None  # the live arm is the simplest existing System 1
    carried = keydoor.make_life("copy", protocol, 0, genes)
    assert carried.brain.efference is not None and carried.brain.arousal is not None
    assert carried.brain.efference.amplitude == protocol["copy"]["efference_amplitude"]
    assert carried.brain.basal_ganglia.config == life.brain.basal_ganglia.config
    recurrent = keydoor.make_life("recurrent", protocol, 0, genes)
    assert recurrent.alpha == protocol["recurrent"]["alpha"] and recurrent.h.shape == (16,)


def test_the_controls_bracket_the_task(quick):
    arms = ("random", "frozen", "step", "tabular", "yoked", "lambda-zero", "blind", "recurrent")
    rows = {arm: keydoor.run_life(arm, 0, 2, quick) for arm in arms}
    for arm, row in rows.items():
        assert "error" not in row and len(row["phases"]) == 3, arm
    recurrent = rows["recurrent"]
    assert recurrent["work"]["brains"] == 0 and recurrent["phases"][0]["aroused"] == 1.0
    assert 0.0 < recurrent["phases"][0]["executed_probability"] < 1.0  # it samples its policy
    for phase in rows["random"]["phases"]:
        assert 0.05 < phase["fed"] < 0.6 and phase["executed_probability"] == 0.5
    frozen = rows["frozen"]
    assert frozen["phases"][1]["executed_probability"] == 1.0  # greedy, with certainty
    assert frozen["phases"][1]["aroused"] == 0.0 and frozen["phases"][0]["aroused"] > 0.0
    assert frozen["work"]["learning_sweeps"] > 0  # it learned under rule A only
    step = rows["step"]["phases"][1]
    assert step["behaviour_interact"] == step["policy_interact"]  # it samples its policy
    assert step["executed_probability"] < 1.0 and step["aroused"] == 1.0
    table = rows["tabular"]
    assert table["work"]["brains"] == 0 and table["work"]["probes"] == 0
    assert rows["blind"]["phases"][0]["visits"] != rows["yoked"]["phases"][0]["visits"]


def test_the_probes_do_not_disturb_the_life_they_read(quick):
    executed = ("fed", "took", "wrong", "lag", "visits", "aroused", "behaviour_interact")
    measurement = (
        "probes",
        "probe_calls",
        "probe_sweeps",
        "probe_memory_reads",
        "checkpoints",
        "latency_ms",
    )
    unprobed = {**quick, "probe_every": 10_000}
    for arm in ("live", "step"):
        probed, plain = (keydoor.run_life(arm, 5, 2, p) for p in (quick, unprobed))
        for a, b in zip(probed["phases"], plain["phases"], strict=True):
            assert [a[key] for key in executed] == [b[key] for key in executed]
        lived = {k: v for k, v in probed["work"].items() if k not in measurement}
        assert lived == {k: v for k, v in plain["work"].items() if k not in measurement}
        assert probed["work"]["probes"] > plain["work"]["probes"]


def test_a_life_saved_in_the_delay_with_its_outcome_pending_continues_identically(
    protocol, tmp_path
):
    """Save between taking the key and reaching the door, the preceding action's outcome
    still to come: the twin lives on identically and takes that outcome once."""
    warnings.simplefilter("ignore")
    cost = protocol["cost"]

    def world(kind, holding, action):
        if action != keydoor.INTERACT:
            return holding, 0.0
        if kind == keydoor.CHEST and not holding:
            return True, 0.0
        if kind == keydoor.DOOR:
            return holding, 1.0 if holding else 0.0
        return holding, 0.0 if kind == keydoor.FLOOR else -cost

    life = keydoor.make_life("live", protocol, 2, protocol["arousal"])
    rng = np.random.default_rng(99)
    pending = (None, False)
    for _ in range(12):  # twelve trips under rule A
        holding = False
        cells = keydoor.corridor(5, protocol["length"], 0, rng)
        for i, kind in enumerate(cells):
            action, _ = life.act(kind, holding, *pending)
            holding, outcome = world(kind, holding, action)
            pending = (outcome, i == len(cells) - 1)
    cells = keydoor.corridor(5, protocol["length"], 0, rng)
    first_lever = cells.index(keydoor.LEVER)
    holding = False
    for kind in cells[:first_lever]:  # the thirteenth trip up to the lamp
        action, _ = life.act(kind, holding, *pending)
        holding, outcome = world(kind, holding, action)
        pending = (outcome, False)
    twin = keydoor.BrainLife(
        cd.Brain.load(life.brain.save(tmp_path / "life.npz")), use_live=True, pouch=True
    )
    mine, theirs = [], []
    for kind in cells[first_lever:] + keydoor.corridor(5, protocol["length"], 0, rng):
        a, _ = life.act(kind, holding, *pending)
        b, _ = twin.act(kind, holding, *pending)
        mine.append(a)
        theirs.append(b)
        holding, outcome = world(kind, holding, a)
        pending = (outcome, kind == keydoor.DOOR)
    assert mine == theirs
    assert twin.brain.arousal.to_dict() == life.brain.arousal.to_dict()


def test_a_refused_answer_is_charged_and_the_crashed_life_keeps_its_ledger(quick, monkeypatch):
    made = keydoor.make_life

    def strict(arm, protocol, seed, genes):
        life = made(arm, protocol, seed, genes)
        learner = life.brain.learner
        learner.config = replace(learner.config, free_steps=48, tolerance=1e-15)
        return life

    monkeypatch.setattr(keydoor, "make_life", strict)
    row = keydoor.run_life("live", 0, 2, {**quick, "episodes": 20})
    assert "error" in row and "did not settle" in row["error"] and row["completed_phases"] == 0
    # the budget and the rescue of as many before the answer is refused
    assert row["work"]["refused_sweeps"] == 96 and row["work"]["brains"] == 1
    assert "phases" not in row and row["work"]["latency_ms"]["aroused"] is None


def test_gates_pool_the_gated_delays_and_count_a_crash_as_a_failure(protocol):
    def life(delay, fed, wrong=0.0, late=0.0, lag=0):
        phases = [{"fed": f, "wrong": wrong, "aroused_late": late, "lag": lag} for f in fed]
        return {"arm": "live", "seed": 0, "delay": delay, "phases": phases}

    good = [life(d, (1.0, 0.95, 1.0)) for d in protocol["gates"]["delays"] for _ in range(5)]
    report = keydoor.gates(good, protocol)
    assert report["passed"] and report["pooled"]["lives"] == 10
    wasteful = good[:-2] + [life(2, (1.0, 1.0, 1.0), wrong=2.0)] * 2  # above the 1.5 of the gate
    report = keydoor.gates(wasteful, protocol)
    assert not report["passed"] and report["pooled"]["frugal"] == pytest.approx(0.8)
    one_restless = good[:-1] + [life(2, (1.0, 1.0, 1.0), late=1.0)]
    assert keydoor.gates(one_restless, protocol)["passed"]  # one life in ten is within the share
    slow_return = good[:-2] + [life(2, (1.0, 1.0, 1.0), lag=51)] * 2  # found later than the gate
    report = keydoor.gates(slow_return, protocol)
    assert not report["passed"] and report["pooled"]["retained"] == pytest.approx(0.8)
    crashed = good + [{"arm": "live", "seed": 9, "delay": 2, "error": "RuntimeError"}]
    report = keydoor.gates(crashed, protocol)
    assert not report["passed"] and report["pooled"]["crashed"] == 1
    ungated = [life(10, (0.5, 0.5, 0.5))]
    assert "passed" not in keydoor.gates(ungated, protocol)  # delay 10 carries no gate


def test_the_second_freezes_receipt_is_its_frozen_protocols_and_carries_its_gates(protocol_1):
    path = HERE / "results" / "confirmation-2026-10-06.json.gz"
    valid, reason = keydoor.verify(path)
    assert valid, reason
    assert "legacy key-door/1" in reason
    body = keydoor.read_receipt(path)
    assert body["frozen_protocol"] and body["genes_override"] is None
    frozen = keydoor.hashlib.sha256((HERE / "protocol.json").read_bytes()).hexdigest()
    assert body["protocol_sha256"] == frozen
    assert body["seeds"] == protocol_1["seeds"]["confirmation"] and body["arms"] == list(
        LEGACY_ARMS
    )
    assert body["delays"] == protocol_1["delays"]
    assert body["gates"] == keydoor.gates(body["rows"], protocol_1)
    assert len(body["rows"]) == len(LEGACY_ARMS) * len(protocol_1["delays"]) * 10


def test_the_first_freezes_receipts_verify_by_their_own_kind():
    """Bound to the sources of commit 35fcb14, they verify without ``--current``."""
    for name in ("confirmation", "variant-scarcity", "variant-composed-critic"):
        path = HERE / "results" / f"freeze1-{name}-2026-10-06.json.gz"
        valid, reason = keydoor.verify(path)
        assert valid, (name, reason)
        body = keydoor.read_receipt(path)
        assert body["protocol_sha256"].startswith("2f214aae")
        assert body["seeds"] == list(range(700, 710))
    first = keydoor.read_receipt(HERE / "results" / "freeze1-confirmation-2026-10-06.json.gz")
    assert first["frozen_protocol"] and not first["gates"]["passed"]
    pooled = first["gates"]["pooled"]
    shares = (pooled["acquired"], pooled["adapted"], pooled["frugal"], pooled["calm"])
    assert shares == (0.9, 1.0, 0.55, 0.7)
    yoked = [r for r in first["rows"] if r["arm"] == "yoked"]
    assert sum("error" in r for r in yoked) == 28  # the fault of that control, repaired since


def test_a_receipt_binds_its_sources_and_refuses_changed_rows(tmp_path, capsys):
    path = tmp_path / "receipt.json.gz"
    run = ["--arms", "tabular", "random", "--seeds", "0", "--delays", "2", "--episodes", "30"]
    run += ["--workers", "1", "--out", str(path)]
    assert keydoor.main(run) == 0
    agree = "canonical form, digest, sources, arithmetic agree"
    assert keydoor.verify(path, current=True) == (True, agree)
    assert keydoor.main(["--verify", str(path)]) == 0 and keydoor.main(["--report", str(path)]) == 0
    stored = json.loads(gzip.decompress(path.read_bytes()))
    body = stored["body"]
    assert stored["kind"] == keydoor.SCHEMA and not body["frozen_protocol"]  # trips overridden
    assert keydoor.read_receipt(path) == body
    files = [item["path"] for item in stored["source"]["files"]]
    assert files[0] == "key_door.py" and "cadence/arousal.py" in files

    def rewrite(content):
        path.write_bytes(gzip.compress((keydoor.canonical_json(content) + "\n").encode()))

    edited = json.loads(json.dumps(stored))
    edited["body"]["rows"][0]["phases"][0]["fed"] = 2.0
    rewrite(edited)
    assert keydoor.verify(path) == (False, "embedded digest does not verify")
    fewer = keydoor.Receipt.build(
        keydoor.SCHEMA, {**body, "rows": body["rows"][:-1]}, keydoor.sources()
    )
    rewrite(fewer.to_dict())
    assert keydoor.verify(path)[1] == "the rows are not the planned lives, each once and in order"
    claimed = keydoor.Receipt.build(
        keydoor.SCHEMA, {**body, "gates": {"passed": True}}, keydoor.sources()
    )
    rewrite(claimed.to_dict())
    assert keydoor.verify(path)[1] == "the stored gates do not follow from the rows"
    tampered = json.loads(json.dumps(body))
    tampered["rows"][0]["phases"][1]["fed"] = 1.5
    rewrite(keydoor.Receipt.build(keydoor.SCHEMA, tampered, keydoor.sources()).to_dict())
    assert keydoor.verify(path)[1] == "a recorded share is outside [0, 1]"
    capsys.readouterr()


def test_world_schedules_do_not_depend_on_actions_or_yoked_relocation(quick, monkeypatch):
    schedules = []

    class Scripted(keydoor.Random):
        def act(self, kind, holding, reward, done):
            schedules[-1].append((kind, done))
            return _selected, True

    monkeypatch.setattr(keydoor, "make_life", lambda *args: Scripted(0))
    for _selected, arm in (
        (keydoor.PASS, "random"),
        (keydoor.INTERACT, "random"),
        (keydoor.INTERACT, "yoked"),
    ):
        schedules.append([])
        row = keydoor.run_life(arm, 0, 2, {**quick, "food": 0.5, "truncation": 0.5})
        assert "error" not in row
    assert schedules[0] == schedules[1] == schedules[2]


def test_door_reading_uses_the_last_fifty_completed_trips_and_probes_the_end(quick, monkeypatch):
    class Scripted(keydoor.Random):
        trips = 0
        probes = []

        def act(self, kind, holding, reward, done):
            action = int(kind == keydoor.DOOR or (kind == keydoor.CHEST and self.trips < 5))
            self.trips += int(kind == keydoor.DOOR)
            return action, True

        def probe(self):
            self.probes.append(self.trips)
            return super().probe()

    life = Scripted(0)
    monkeypatch.setattr(keydoor, "make_life", lambda *args: life)
    row = keydoor.run_life("random", 0, 2, {**quick, "truncation": 0})
    assert row["phases"][0]["opened"] is None  # early key visits are outside the window
    assert life.probes == [0, 20, 40, 60, 60, 80, 100, 120, 120, 140, 160, 180]
    assert row["pending_outcome"] == {"reward": 0.0, "done": True}
    assert row["yoked_bank"] == 0.0


def test_no_completed_trips_are_null_readings_and_cannot_pass_gates(quick):
    row = keydoor.run_life("random", 0, 2, {**quick, "truncation": 1.0})
    for phase in row["phases"]:
        assert phase["episodes"] == 0 and phase["cut"] == quick["episodes"]
        assert all(phase[k] is None for k in ("fed", "fed_whole", "took", "wrong", "opened", "lag"))
    assert not keydoor.gates([{**row, "arm": "live"}], quick)["passed"]
    assert "no completed trips" in keydoor.summarize([row])
    keydoor.canonical_json(row)


def test_cli_can_write_verify_and_report_a_life_with_no_completed_trips(quick, tmp_path, capsys):
    source = tmp_path / "protocol.json"
    source.write_text(json.dumps({**quick, "truncation": 1.0, "episodes": 1}))
    output = tmp_path / "receipt.json.gz"
    assert (
        keydoor.main(
            [
                "--protocol",
                str(source),
                "--arms",
                "random",
                "--seeds",
                "0",
                "--delays",
                "2",
                "--workers",
                "1",
                "--out",
                str(output),
            ]
        )
        == 0
    )
    assert keydoor.verify(output, current=True, protocol=source)[0]
    assert keydoor.main(["--report", str(output)]) == 0
    capsys.readouterr()


def test_food_availability_does_not_shift_when_an_arm_skips_early_meals(quick, monkeypatch):
    class Scripted(keydoor.Random):
        def __init__(self, skip):
            super().__init__(0)
            self.skip, self.trips, self.rewards = skip, 0, []

        def act(self, kind, holding, reward, done):
            if done:
                self.rewards.append(reward)
            action = int(
                self.trips >= self.skip and kind in (keydoor.CHEST, keydoor.LAMP, keydoor.DOOR)
            )
            self.trips += int(kind == keydoor.DOOR)
            return action, True

    histories = []
    for skip in (0, 20):
        life = Scripted(skip)
        monkeypatch.setattr(keydoor, "make_life", lambda *args, life=life: life)
        row = keydoor.run_life("random", 0, 2, {**quick, "food": 0.5, "truncation": 0})
        assert "error" not in row
        histories.append(life.rewards + [row["pending_outcome"]["reward"]])
    assert histories[0][20:] == histories[1][20:]
    assert set(histories[0][20:]) == {0.0, 1.0}


def test_each_delivered_reward_belongs_to_the_preceding_executed_action(quick, monkeypatch):
    recorded = []

    class Scripted(keydoor.Random):
        def act(self, kind, holding, reward, done):
            action = int(self.rng.integers(2))
            recorded.append((kind, holding, reward, done, action))
            return action, True

    monkeypatch.setattr(keydoor, "make_life", lambda *args: Scripted(11))
    row = keydoor.run_life("random", 0, 2, {**quick, "episodes": 5, "truncation": 0})
    expected = []
    rules = [keydoor.KINDS.index(name) for name in quick["rules"]]
    for i, (kind, holding, _reward, _done, action) in enumerate(recorded):
        keyed = rules[min(len(rules) - 1, i // (5 * quick["length"]))]
        outcome = 0.0
        if action == keydoor.INTERACT:
            if kind == keydoor.DOOR:
                outcome = float(holding)
            elif kind != keydoor.FLOOR and not (kind == keyed and not holding):
                outcome = -quick["cost"]
        expected.append(outcome)
    assert [r[2] for r in recorded] == [None, *expected[:-1]]
    assert row["pending_outcome"]["reward"] == expected[-1]


@pytest.mark.parametrize("changes", [{"arms": []}, {"seeds": [0, 0]}, {"delays": [12]}])
def test_invalid_plans_are_refused_before_running(quick, changes):
    with pytest.raises(ValueError):
        keydoor.run(quick, **{"arms": ["random"], "seeds": [0], "delays": [2], **changes})


def test_receipt_rejects_self_signed_inconsistent_manifest_phases_and_work(tmp_path, capsys):
    path = tmp_path / "receipt.json.gz"
    assert (
        keydoor.main(
            [
                "--arms",
                "random",
                "--seeds",
                "0",
                "--episodes",
                "30",
                "--workers",
                "1",
                "--out",
                str(path),
            ]
        )
        == 0
    )
    stored = json.loads(gzip.decompress(path.read_bytes()))

    def check_edit(edit, message):
        edited = json.loads(json.dumps(stored))
        edit(edited)
        edited["digest"] = keydoor.canonical_sha256(
            {k: edited[k] for k in ("kind", "body", "source")}
        )
        path.write_bytes(gzip.compress((keydoor.canonical_json(edited) + "\n").encode()))
        assert message in keydoor.verify(path)[1]
        assert keydoor.main(["--report", str(path)]) == 1

    check_edit(lambda s: s["source"].update(manifest_sha256="0" * 64), "source manifest digest")

    def swap_rules(s):
        phases = s["body"]["rows"][0]["phases"]
        phases[0], phases[1] = phases[1], phases[0]

    check_edit(swap_rules, "the protocol's rules")
    check_edit(lambda s: s["body"]["rows"][0]["work"].update(learning_sweeps=-1), "work counters")
    check_edit(lambda s: s["body"].update(arms=[], rows=[], gates={}), "nonempty and unique")
    capsys.readouterr()


@pytest.mark.parametrize("arm", ["live", "step"])
def test_feedback_work_is_charged_when_the_following_action_refuses(protocol, monkeypatch, arm):
    life = keydoor.make_life(arm, protocol, 0, protocol["arousal"])
    life.act(keydoor.FLOOR, False, None, False)
    before = life.work["learning_sweeps"]
    original = life.brain._settled

    def refuse(*args, **kwargs):
        life.brain.learner.config = replace(
            life.brain.learner.config, free_steps=1, tolerance=1e-15
        )
        return original(*args, **kwargs)

    monkeypatch.setattr(life.brain, "_settled", refuse)
    with pytest.raises(RuntimeError, match="did not settle"):
        life.act(keydoor.CHEST, False, 0.0, False)
    assert life.brain.last_learning["free_steps"] > 0
    assert life.work["learning_sweeps"] == before + life.brain.last_learning["free_steps"]
    assert life.work["refused_sweeps"] == 2  # the budget and its rescue


def test_a_completed_routine_forecast_is_charged_if_the_woken_answer_refuses(
    protocol, monkeypatch, tmp_path
):
    genes = {**protocol["arousal"], "youth": 0, "threshold": 0.05}
    life = keydoor.make_life("live", protocol, 0, genes)
    life.act(keydoor.FLOOR, False, None, False)
    assert life.work["routine"] == 1 and not life.brain.arousal.aroused
    twin = cd.Brain.load(life.brain.save(tmp_path / "before.npz"))
    forecast_steps = []

    # Both brains run the same real forecast, accept the waking outcome and then refuse
    # the next answer. The independent twin records the forecast's completed work.
    forecast = twin._forecast

    def record_forecast(*args, **kwargs):
        answer = forecast(*args, **kwargs)
        forecast_steps.append(twin.last_settlement["steps"])
        return answer

    monkeypatch.setattr(twin, "_forecast", record_forecast)
    for brain in (life.brain, twin):
        original = brain._settled

        def refuse(*args, brain=brain, original=original, **kwargs):
            # The shared settlement path also serves the preceding routine forecast.
            # Only the answer after accepting the waking outcome must refuse.
            if brain.arousal.aroused:
                # One sweep from rest at an impossible tolerance: a cached forecast
                # state can sit at an exact fixed point and would otherwise pass.
                brain.learner.config = replace(
                    brain.learner.config, free_steps=1, tolerance=1e-15
                )
                brain.basal_ganglia._free = None
            return original(*args, **kwargs)

        monkeypatch.setattr(brain, "_settled", refuse)
    # a punishment no calm forecast allowed for wakes either brain for certain
    with pytest.raises(RuntimeError, match="did not settle"):
        life.act(keydoor.CHEST, False, -1.0, False)
    with pytest.raises(RuntimeError, match="did not settle"):
        twin.live(keydoor.observe(keydoor.CHEST, False, True), reward=[-1.0], done=[False])
    assert life.brain.arousal.aroused and life.brain.hippocampus.writes == 1
    assert len(forecast_steps) == 1 and forecast_steps[0] > 0
    assert life.work["aborted_forecast_sweeps"] == sum(forecast_steps)
    assert life.work["refused_sweeps"] == 2  # the budget and its rescue
    assert life._forecast_sweeps == 0
    with (
        np.load(life.brain.save(tmp_path / "after.npz")) as actual,
        np.load(twin.save(tmp_path / "twin.npz")) as expected,
    ):
        assert actual.files == expected.files
        for name in actual.files:
            np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)

    # A retry without another outcome must not count the preceding forecast twice.
    with pytest.raises(RuntimeError, match="did not settle"):
        life.act(keydoor.CHEST, False, None, False)
    assert life.work["aborted_forecast_sweeps"] == sum(forecast_steps)
    assert life.work["refused_sweeps"] == 4  # two refusals, each the budget and its rescue
