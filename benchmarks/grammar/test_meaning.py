"""Where a sentence's meaning lives (cadence#286): the erasures, the probes, the controls'
stepped speakers and a short diagnosis life."""

from __future__ import annotations

import copy
import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


gr = _load("grammar")
mn = _load("meaning")


def _heard(brain, rows):
    brain.reset()
    for t in range(len(rows[0].pre)):
        brain.act(np.stack([r.pre[t] for r in rows]), greedy=True)


def test_the_default_composition_is_what_compose_gives():
    warnings.simplefilter("ignore")
    brain = mn.make_brain(0)
    assert brain.recall is not None and brain.recall.anchor and brain.recall.drive_gain
    assert brain.learner.config.nudged_fixed and brain.hearing is not None


def test_state_only_empties_every_slot_written_before_go():
    warnings.simplefilter("ignore")
    brain = mn.make_brain(0)
    rng = np.random.default_rng(0)
    rows = [gr.clause(rng, 1) for _ in range(4)]
    _heard(brain, rows)
    assert np.abs(brain.recall.values).sum() > 0
    mn.intervention("state")(brain, rows)
    for name in ("keys", "values", "nexts"):
        assert not np.asarray(getattr(brain.recall, name)).any()
    assert brain.working_memory.trace.any()  # the carried state stays
    mn.intervention("neither")(brain, rows)
    assert not brain.working_memory.trace.any()


def test_a_paths_start_and_its_facts_are_erased_apart():
    warnings.simplefilter("ignore")
    brain = mn.make_brain(0)
    rng = np.random.default_rng(1)
    rows = [gr.path(rng, 2) for _ in range(4)]
    _heard(brain, rows)
    start = np.stack([r.pre[-1] for r in rows])
    fork = copy.deepcopy(brain)
    mn.intervention("start")(fork, rows)
    assert not fork.recall.values[:, -1].any() and not fork.recall.nexts[:, -2].any()
    assert fork.recall.values[:, :-1].any()  # the facts stay
    mn.intervention("facts")(brain, rows)
    assert np.allclose(brain.recall.values[:, -1], start)
    assert not brain.recall.values[:, :-1].any()
    assert not brain.recall.nexts[:, :-1].any()  # an emptied slot holds no next either


def test_a_silent_read_drives_nothing_and_is_heard_again():
    warnings.simplefilter("ignore")
    brain = mn.make_brain(0)
    rng = np.random.default_rng(2)
    rows = [gr.clause(rng, 0) for _ in range(4)]
    _heard(brain, rows)
    ports = np.concatenate([np.asarray(brain.connectome.populations[n])
                            for n in ("recall", "next")])
    go = np.tile(gr.concept("GO"), (4, 1))
    assert np.abs(brain.stimulus(go)[:, ports]).sum() > 0
    mn.intervention("silent")(brain, rows)
    assert not brain.stimulus(go)[:, ports].any()


def test_the_components_of_each_family():
    rng = np.random.default_rng(3)
    row = gr.clause(rng, 2, agent=(1, 1), patient=(2, 0), v=1, mods=[(0, 0), (3, 1)])
    assert mn.components(row) == {
        "agent_noun": 1, "agent_number": 1, "verb": 1, "patient_noun": 2, "patient_number": 0,
        "distractors": 2, "distractor1_noun": 0, "distractor1_number": 0,
        "distractor2_noun": 3, "distractor2_number": 1, "verb_form": 3}
    assert mn.positions(row) == {"go": 0, "verb": 5, "end": 7}
    row = gr.dyck(rng, [1, 1, 0, 1], [0, 1, 0, 1])  # ( [ ] [  leaves ( [ open
    assert mn.components(row) == {"open": 1, "innermost": 1, "outermost": 0,
                                  "second_innermost": 0}
    row = gr.path(rng, 2)
    parts = mn.components(row)
    assert parts["start"] == row.target[0] and parts["first_hop"] == row.target[2]
    assert parts["hops"] == 1 and mn.positions(row) == {"go": 0, "end": len(row.target) - 1}


def test_a_probe_reads_a_linear_code_and_reads_chance_on_noise():
    rng = np.random.default_rng(4)
    y = rng.integers(4, size=600)
    code = rng.normal(size=(4, 12))
    x = code[y] + 0.3 * rng.normal(size=(600, 12))
    held = np.arange(600) % 3 == 0
    found = mn.fit_probe(x, y, held)
    assert found["accuracy"] > 0.95 and found["n"] == 200
    noise = mn.fit_probe(rng.normal(size=(600, 12)), y, held)
    assert noise["accuracy"] < 0.45
    assert np.isnan(mn.fit_probe(x, np.zeros(600, dtype=int), held)["accuracy"])


def test_the_probe_sentences_are_the_founders_and_at_trained_shapes():
    a, b = mn.probe_batches(3, 0.05), mn.probe_batches(3, 0.05)
    assert [[r.target for r in rows] for rows in a["clause"]] == [
        [r.target for r in rows] for rows in b["clause"]]
    assert {r.meta["L"] for rows in a["clause"] for r in rows} == set(gr.TRAIN_L)
    assert all(r.meta["depth"] <= gr.TRAIN_DEPTH for rows in a["dyck"] for r in rows)
    assert {r.meta["hops"] for rows in a["path"] for r in rows} == set(gr.TRAIN_HOPS)


def test_the_stepped_speakers_say_what_the_controls_say():
    pytest.importorskip("torch")
    ct = mn._controls()
    tests = gr.panels(0, 1)
    for model in ("lstm", "transformer"):
        learner = ct.Learner(model, 16, 0, 0.003)
        learner.net.eval()
        for name in ("clause", "ctx_agent", "path"):
            rows = tests[name][0]
            stepped = (mn.lstm_speak(learner.net, rows) if model == "lstm"
                       else mn.transformer_speak(learner.net, rows))
            assert stepped == learner.speak(rows)


def test_the_controls_floors_say_nothing_of_the_meaning():
    pytest.importorskip("torch")
    ct = mn._controls()
    rng = np.random.default_rng(5)
    a = [gr.clause(rng, 1) for _ in range(4)]
    b = [gr.clause(rng, 1) for _ in range(4)]
    lstm = ct.Learner("lstm", 16, 0, 0.003).net.eval()
    assert mn.lstm_speak(lstm, a, erase=True) == mn.lstm_speak(lstm, b, erase=True)
    transformer = ct.Learner("transformer", 16, 0, 0.003).net.eval()
    assert mn.transformer_speak(transformer, a, masked=True) == mn.transformer_speak(
        transformer, b, masked=True)
    states = mn.lstm_states(lstm, a)
    assert states.shape == (4, len(a[0].pre) + len(a[0].target), 32)


def test_a_short_diagnosis_life():
    pytest.importorskip("torch")
    row = mn.where_life({"founder": 0, "batches": 2, "panel_batches": 1, "probe_scale": 0.05,
                         "controls": ["lstm"]})
    assert set(row["sentences"]) == set(mn.CONDITIONS + mn.PATH_CONDITIONS) | {"context"}
    assert set(row["sentences"]["state"]) == set(mn.WHOLE + mn.LONG)
    assert set(row["sentences"]["start"]) == {"path", "path_long"}
    assert set(row["probes"]) == {"intact", "state"}
    found = row["probes"]["intact"]["clause"]["agent_noun"]["go"]
    assert set(found) == set(mn.PLACES) and 0.0 <= found["read"]["accuracy"] <= 1.0
    # with the slots emptied the read holds nothing at GO: its probe reads chance
    blind = row["probes"]["state"]["clause"]["agent_noun"]["go"]["read"]
    assert blind["accuracy"] == blind["chance"]
    control = row["controls"][0]
    assert control["model"] == "lstm" and control["stepped_agrees"]
    assert set(control["panels"]) == {"intact", "floor"}
    assert "carried" in control["probes"]["clause"]["verb"]["go"]
    assert mn.read([row])[0].startswith("lives: 1")


# ------------------------------------------------------------------ the message, emulated


def _heard_into(message, rows):
    message.reset(len(rows))
    for t in range(len(rows[0].pre)):
        message.hear(np.stack([r.pre[t] for r in rows]))


def test_the_instruments_phased_message_is_the_librarys():
    import cadence as cd

    rng = np.random.default_rng(0)
    rows = [gr.clause(rng, 2) for _ in range(6)]
    emulated = mn.Message({"code": "separator", "blocks": ["phase:4"], "scales": [1.0]}, 3)
    _heard_into(emulated, rows)
    library = cd.Message(gr.INPUTS, phases=4, units=70, winners=5, fan_in=16, seed=3)
    for t in range(len(rows[0].pre)):
        library.hear(np.stack([r.pre[t] for r in rows]))
    assert np.array_equal(emulated.state(scaled=False), library.state)


def test_the_blocks_hold_an_episode_each_in_its_own_way():
    rng = np.random.default_rng(1)
    rows = [gr.clause(rng, 1) for _ in range(4)]
    codes = mn.Message({"blocks": ["bag"]}, 0)
    each = [codes.code(np.stack([r.pre[t] for r in rows])) for t in range(4)]
    for blocks, want in (
        (["bag"], sum(each)),
        (["set"], np.maximum.reduce(each)),
        (["primacy:0.5"], sum(0.5 ** t * c for t, c in enumerate(each))),
        (["recent:0.5"], sum(0.5 ** (3 - t) * c for t, c in enumerate(each))),
        (["phase:2"], np.concatenate([np.maximum(each[0], each[2]),
                                      np.maximum(each[1], each[3])], axis=1)),
        (["slots:2"], np.concatenate([each[3], each[2]], axis=1)),
    ):
        message = mn.Message({"blocks": blocks, "scales": [1.0]}, 0)
        _heard_into(message, rows)
        assert np.allclose(message.state(), want), blocks
    register = mn.Message({"blocks": ["register:2"], "scales": [1.0]}, 0)
    _heard_into(register, rows)
    assert np.array_equal(register.state()[:, :gr.INPUTS], np.stack([r.pre[3] for r in rows]))
    assert mn.carrier_blocks({"blocks": ["phase:4"]}) == 8 and mn.carrier_blocks(None) == 0
    held = mn.Message({"hold": True}, 0)
    _heard_into(held, rows)
    before = held.state().copy()
    held.hear(np.tile(gr.concept("GO"), (4, 1)), speaking=True)
    assert np.array_equal(held.state(), before)


def test_the_shams_drive_the_same_neurons_without_the_meaning():
    rng = np.random.default_rng(2)
    rows = [gr.clause(rng, 1) for _ in range(4)]
    design = {"code": "separator", "blocks": ["phase:4"], "scales": [1.0]}
    real, present, shuffled = (mn.Message({**design, **extra}, 0) for extra in (
        {}, {"sham": "present"}, {"sham": "shuffled"}))
    for message in (real, present, shuffled):
        _heard_into(message, rows)
    assert present.state().sum(axis=1).tolist() == [5.0] * 4  # the last moment's code alone
    assert np.array_equal(shuffled.state(), np.roll(real.state(), 1, axis=0))


def test_the_message_is_installed_on_its_carrier_and_the_bar_spares_the_given_words():
    warnings.simplefilter("ignore")
    design = {"code": "separator", "blocks": ["phase:4"], "scales": [0.5]}
    brain = mn.message_brain(0, design)
    message = mn.Message(design, 0)
    rows = gr.panels(0, 1)["ctx_agent"][0]  # the agent and one distractor are given
    assert rows[0].force == 3
    seen = {}

    def look(b, rs):
        seen["values"] = np.array(b.recall.values)

    mn.speak(brain, rows, teach=False, message=message, bar=True, intervene=look)
    carried = brain.history.trace
    assert carried.shape == (len(rows), 280) and set(np.unique(carried)) <= {0.0, 0.5}
    heard = len(rows[0].pre)
    values = seen["values"]  # at the continuation's start: a boundary, the meaning, GO, 2 words
    assert values.shape[1] == 1 + heard + rows[0].force
    assert not values[:, :1 + heard].any() and values[:, 1 + heard:].any(axis=2).all()


def test_note_targets_fill_as_the_meaning_is_heard():
    rng = np.random.default_rng(0)
    row = gr.clause(rng, 2, agent=(1, 1), patient=(2, 0), v=1, mods=[(0, 0), (3, 1)],
                    inverted=True)
    assert [mn.note_targets(row, h) for h in range(6)] == [
        [8, 2, 8, 8, 8], [8, 1, 8, 8, 8], [8, 1, 2, 8, 8], [5, 1, 2, 8, 8], [5, 1, 2, 0, 8],
        [5, 1, 2, 0, 7]]
    assert mn.note_targets(gr.path(rng, 1), 9) == [8, 2, 8, 8, 8]
    assert mn.note_sizes() == (19, 9, 3, 9, 9, 9)


def test_retrieval_by_the_code():
    found = mn.retrieval(0, stored=(16,), cues=8)["16"]
    assert found["reordered"]["key"] == 1.0 and found["swapped"]["key"] == 1.0
    assert found["reordered"]["state"] < 1.0  # the phased state keeps the order heard
    assert found["crossed"]["key"] <= 0.5  # words and concepts give unrelated keys
    assert set(found) == {"reordered", "swapped", "partial", "words", "crossed"}


def test_short_lives_of_every_kind():
    warnings.simplefilter("ignore")
    design = {"code": "separator", "blocks": ["phase:4"], "scales": [0.15]}
    row = mn.message_life({"arm": "message", "message": design, "founder": 0, "batches": 2,
                           "panel_batches": 1, "probe_scale": 0.05, "twins": True,
                           "invariance": True, "after": [["g2", 1]]})
    assert set(row["panels"]) == set(mn.WHOLE + mn.LONG) and "panels_barred" in row
    assert row["invariance"]["key"]["same"] == 1.0 and row["invariance"]["n"] == 64
    assert "ctx_verb_erased_message" in row["twins"] and row["persistence"][0]["grammar"] == "g2"
    assert "message" in row["probes"]["clause"]["verb"]["go"]
    assert mn.macs(row) > 0
    barred = mn.message_life({"arm": "message_barred", "message": design, "founder": 0,
                              "batches": 2, "panel_batches": 1, "train": "capacity"})
    assert set(barred["capacity"]) == {"L0", "L1", "L2", "L3", "L4"} and barred["bar"] == 1.0
    notes = mn.notes_life({"founder": 0, "batches": 2, "panel_batches": 1})
    assert set(notes["notes"]["clause"]) == {"all_at_go", "all_at_end", *mn.NOTES}
    lines = mn.read_lives([row, barred, notes])
    assert any(line.startswith("capacity") for line in lines)
    assert mn.passes([0.95, 0.91, 0.5, 0.9, 0.92, 0.99], 0.90) == (5, True)


# ----------------------------------------------------------------------- the declared run


def test_a_paraphrase_hears_the_first_grammar_and_says_the_second():
    rng = np.random.default_rng(0)
    base = gr.clause(rng, 1, agent=(0, 1), patient=(2, 0), v=0, mods=[(3, 0)])
    row = mn.paraphrase(base)
    heard = [int(np.argmax(x[:gr.ACTIONS])) for x in row.pre[:-1]]
    assert gr.say(heard) == "dogs near owl see fox end"
    assert row.pre[-1][gr.C["DIALECT"]] == 1.0 and not row.pre[0][gr.C["SPEAK"]]
    assert gr.say(row.target) == "dogs near owl fox see end"
    assert gr.score(row, row.target)["roles"] and row.meta["paraphrase"]
    panels = mn.paraphrase_panels(0, 1)
    split = gr.Split.of(0)
    held = [split.held(tuple(r.meta["agent"]), tuple(r.meta["patient"]), r.meta["verb"])
            for r in panels["para_held"][0]]
    assert all(held) and set(panels) == {"para", "para_held", "g2"}


def test_the_declared_plan_names_every_arm_and_matches_its_controls():
    warnings.simplefilter("ignore")
    plan = mn.declared_plan((9628600,))
    tags = {spec["tag"] for spec in plan["brains"]}
    assert {"default", "barred", "message", "message_wide", "sham", "state", "state_default",
            "restated", "paraphrase", "paraphrase_read"} <= tags
    assert {t for t in tags if t.startswith("capacity")} == {
        "capacity_u70", "capacity_u140", "capacity_u280", "capacity_m64", "capacity_m256"}
    state = next(s for s in plan["brains"] if s["tag"] == "state")
    assert state["message"]["units"] == 280 and state["module"] == 128
    assert state["settings"] == {"learning_surprise": False} and state["after"] == mn.LATER
    sizes = sorted({c["parameters"] for c in plan["controls"]})
    assert sizes[0] == mn.make_brain(9628600).parameters() and len(sizes) == 3
    assert len(plan["controls"]) == 12 and {k["chamber"] for k in plan["knock"]} >= {"battery"}
    assert {s["tag"] for s in plan["amendments"]} == {"heard", "state_256", "wide_256",
                                                     "plain_256"}
    assert all(k["module"] == 256 for k in plan["amendments_knock"])
    assert len(plan["amendments_knock"]) == len(plan["knock"])


def test_a_message_brain_on_a_chambers_own_loop_is_the_instruments_speak():
    warnings.simplefilter("ignore")
    design = {"code": "separator", "blocks": ["phase:4"], "scales": [0.15]}
    a, message = mn.message_brain(1, design), mn.Message(design, 1)
    b = mn.message_class({"units": 70, "scale": 0.15}).compose(
        gr.INPUTS, gr.ACTIONS, modules=(64,), seed=1, backend="cpu", recall=True)
    rng, split = np.random.default_rng(0), gr.Split.of(1)
    for _ in range(3):
        rows = gr.train_batch(rng, split, "g1")
        assert mn.speak(a, rows, teach=True, message=message) == gr.speak(b, rows, teach=True)
    assert np.array_equal(a.brain.efficacy, b.brain.efficacy)
    assert copy.deepcopy(b).held.told.tolist() == b.held.told.tolist()


def test_the_gates_read_a_small_run():
    warnings.simplefilter("ignore")
    plan = mn.declared_plan((0,))
    rows = []
    for spec in plan["brains"]:
        if spec["tag"] not in ("default", "message", "state", "paraphrase", "capacity_u70"):
            continue
        small = {**spec, "batches": 2, "panel_batches": 1}
        if "after" in small:
            small["after"] = [["g2", 1], ["g1", 1]]
        if small.get("probe_scale", 0) > 0:
            small["probe_scale"] = 0.05
        rows.append(mn.run_life(small))
    knock = [mn.run_life({**k, "lessons": 2, "tests": 1}) for k in plan["knock"][:2]]
    lines = mn.verdict(rows, [], knock)
    assert lines[0].startswith("H1") and any(line.startswith("K, other") for line in lines)
    assert any("state clause_held" in line for line in lines)
    assert mn.paired([0.9, 0.8, 1.0], [0.8, 0.8, 0.8])[0] == pytest.approx(0.1)
    found = mn.read_retrieval([{"kind": "retrieval", "founder": 0, "units": 70,
                                "found": mn.retrieval(0, stored=(16,), cues=4)}])
    assert found[0].startswith("retrieval") and "reordered" in found[1]


def test_the_second_plan_sets_a_control_beside_every_message_arm():
    plan = mn.own_plan((0, 1))
    arms: dict[str, dict] = {}
    for spec in plan["brains"]:
        arms.setdefault(spec["tag"], spec)
    assert len(plan["brains"]) == 2 * len(arms)
    # the control is the same life with the previous-word signal in the message's place
    for tag, control in (("state_256", "heard_256"), ("read_64", "heard_64"),
                         ("paraphrase", "paraphrase_heard"),
                         ("capacity_256", "capacity_heard")):
        ours, theirs = arms[tag], arms[control]
        held = {k: v for k, v in ours["settings"].items() if not k.startswith("message")}
        assert theirs["settings"] == {**mn.HEARD, **held}
        for name in ("arm", "module", "train", "batches"):
            assert ours.get(name) == theirs.get(name)
    assert arms["read_64"]["settings"]["message_read"] is True
    assert "message_read" not in arms["direct_64"]["settings"]
    assert arms["message_fixed"]["settings"] == {**arms["message_quiet"]["settings"],
                                                 **arms["default_fixed"]["settings"]}
    # the controls sit at the brains' parameter counts, past the matched width's 256
    sizes = {c["parameters"] for c in plan["controls"]}
    assert len(sizes) == 3 and len(plan["controls"]) == 3 * 2 * 2 * 2
    ct = mn._controls()
    for c in plan["controls"]:
        assert abs(ct.count(ct.build(c["model"], c["width"], 0)) - c["parameters"]) < (
            0.02 * c["parameters"])


def test_the_second_gates_read_a_small_run_of_the_brains_own_message():
    warnings.simplefilter("ignore")
    plan = mn.own_plan((0,))
    rows = []
    for spec in plan["brains"]:
        if spec["tag"] not in ("default", "default_fixed", "state_256", "heard_256", "read_64",
                               "message_quiet", "message_fixed", "paraphrase", "capacity_256"):
            continue
        small = {**spec, "batches": 2, "panel_batches": 1, "module": 64}
        if small.get("probe_scale", 0) > 0:
            small["probe_scale"] = 0.05
        rows.append(mn.run_life(small))
    by = {r["tag"]: r for r in rows}
    assert "panels_silent" in by["message_quiet"] and "panels_barred" in by["message_quiet"]
    assert "panels_silent" not in by["default"] and "located" in by["default"]
    assert "message" in next(iter(by["read_64"]["probes"]["clause"].values()))["go"]
    lines = mn.own_verdict(rows, [])
    assert lines[0].startswith("whole sentences") and "| `state_256` |" in "\n".join(lines)
    assert any(line.startswith("F, nothing barred") for line in lines)
    assert any("message_quiet, silent" in line for line in lines)
    assert any(line.startswith("C, the message less") for line in lines)


def test_the_third_plan_and_its_gates_read_small_lives():
    warnings.simplefilter("ignore")
    plan = mn.third_plan((0, 1), located=(2,))
    assert {s["tag"] for s in plan["free"]} == {"default", "heard", "quiet", "quiet_24", "quiet_32",
                                                "quiet_hold", "quiet_clear", "quiet_10"}
    assert {s["tag"] for s in plan["roles_located"]} == {"own_256", "instrument_256", "fine_256"}
    assert {s["tag"] for s in plan["roles"]} == {"own_256", "fine_256"}
    assert mn.GO_CHANNEL == 33
    rows = []
    for spec in plan["free"]:
        if spec["tag"] in ("default", "heard", "quiet_24", "quiet_hold", "quiet_clear"):
            rows.append(mn.run_life({**spec, "batches": 2, "panel_batches": 1}))
    by = {(r["tag"], r["founder"]): r for r in rows}
    assert by[("quiet_24", 0)]["parameters"] > by[("default", 0)]["parameters"]
    assert "panels_silent" in by[("quiet_hold", 0)]
    lines = mn.third_free_verdict(rows)
    assert lines[0].startswith("whole sentences") and any("-> " in line for line in lines)
    assert any(line.startswith("by founder") for line in lines)
    roles = [mn.run_life({**spec, "batches": 2, "panel_batches": 1, "module": 64})
             for spec in plan["roles_located"]]
    fresh = [mn.run_life({**spec, "batches": 2, "panel_batches": 1, "module": 64})
             for spec in plan["roles"] if spec["founder"] == 0]
    # the trace, the read and 4 message blocks; 32 carrier blocks; 32 groups of the message
    # the default brain hears the moment before (history_decays=(0.0,)): one more carried gain
    assert sorted(len(r["gains"]) for r in roles) == [7, 34, 35]
    lines = mn.third_roles_verdict(roles, fresh, [{**roles[0], 'tag': 'state_256'}])
    assert any(line.startswith('| `state_256 (second run)`') for line in lines)
    assert any(line.startswith("gate, fine_256") for line in lines)
    assert any("less own_256, paired" in line for line in lines)


def test_the_confirmatory_plan_and_its_gate_read_small_lives():
    warnings.simplefilter("ignore")
    plan = mn.third_confirm_plan((0, 1))
    assert [s["tag"] for s in plan[::2]] == ["default", "heard", "quiet_hold", "quiet_hold_heard"]
    assert plan[-1]["settings"] == {**mn.HOLD, **mn.HEARD} and len(mn.CONFIRM) == 12
    rows = [mn.run_life({**spec, "batches": 2, "panel_batches": 1}) for spec in plan]
    lines = mn.third_confirm_verdict(rows)
    assert sum("-> F3c" in line for line in lines) == 3
    assert any(line.startswith("each arm less `heard`") for line in lines)
    assert any(line.startswith("  quiet_hold_heard:") for line in lines)


def test_the_fourth_plan_reads_held_out_combinations_and_their_probes():
    warnings.simplefilter("ignore")
    plan = mn.fourth_plan((0, 1))
    assert [s["tag"] for s in plan[::2]] == ["own_256", "read_256", "heard_own_256", "both_256"]
    assert len(mn.FOURTH) == 12 and all(s["module"] == 256 for s in plan)
    both = mn.split_batches(0, 1)
    split = gr.Split.of(0)
    assert all(split.held(tuple(r.meta["agent"]), tuple(r.meta["patient"]), r.meta["verb"])
               for rows in both["held"] for r in rows)
    assert not any(split.held(tuple(r.meta["agent"]), tuple(r.meta["patient"]), r.meta["verb"])
                   for rows in both["trained"] for r in rows)
    rows = [mn.run_life({**spec, "module": 64, "batches": 2, "panel_batches": 1,
                         "held_probes": 2}) for spec in plan]
    assert "said_held" in rows[0] and "probes_held" in rows[0]
    some = next(iter(rows[0]["probes_held"].values()))
    assert "message" in some["go"]
    lines = mn.fourth_verdict(rows, of=2)
    assert sum("-> H1c" in line for line in lines) == 4
    assert any("Multiply-accumulates" in line for line in lines)


def test_the_long_plan_reads_lengths_and_depths_from_the_state():
    warnings.simplefilter("ignore")
    plan = mn.fourth_long_plan((0,))
    assert [s["tag"] for s in plan["long"]] == ["both_256", "default"]
    assert len(plan["long_controls"]) == 4 and len(mn.FOURTH_LONG) == 6
    assert all(len(r.pre) > 5 for rows in mn.long_batches(0, 1) for r in rows)
    rows = [mn.run_life({**spec, "module": 64, "batches": 2, "panel_batches": 1,
                         "long_probes": 2}) for spec in plan["long"]]
    by = {r["tag"]: r for r in rows}
    assert any(k.startswith("clause_L3") for k in by["both_256"]["said_long"])
    assert "message" in next(iter(by["both_256"]["probes_long"].values()))["go"]
    assert "located" in by["default"]
    lines = mn.fourth_long_verdict(rows, [])
    assert sum("-> A2s" in line for line in lines) == 2
    assert any(line.startswith("  default clause_long:") for line in lines)


def test_the_winners_and_fifth_plans_and_their_gates_read_small_lives():
    warnings.simplefilter("ignore")
    plan = mn.winners_plan((0,))
    assert [s["tag"] for s in plan] == ["both_256", "both_256_w10"]
    assert plan[1]["settings"] == {**mn.BOTH, "message_winners": 10}
    rows = [mn.run_life({**spec, "module": 64, "batches": 2, "panel_batches": 1})
            for spec in plan]
    lines = mn.fourth_verdict(rows, of=1, base="both_256")
    assert any("less both_256" in line for line in lines)
    fifth = mn.fifth_plan((0,))
    assert [s["settings"].get("message_phases", 4) for s in fifth] == [4, 7, 8, 16]
    small = [mn.run_life({**spec, "module": 64, "batches": 2, "panel_batches": 1,
                          "long_probes": 2}) for spec in fifth[:2]]
    lines = mn.fifth_verdict(small, of=1)
    assert sum("-> A2c" in line for line in lines) == 2
    assert any("dyck_d4" in line for line in lines)


def test_the_sixth_plan_keys_the_message_by_role_and_its_gates_read_small_lives():
    warnings.simplefilter("ignore")
    plan = mn.sixth_plan(tuple(range(24)))
    tags = [s["tag"] for s in plan["sixth"]]
    assert tags.count("both_256") == 24 and tags.count("both_p8") == 12
    assert tags.count("role_256") == 24 and len(plan["sixth_controls"]) == 2 * 2 * 2 * 6
    role = next(s for s in plan["sixth"] if s["tag"] == "role_256")
    brain = mn.make_brain(0, role["settings"], 64)
    assert brain.message.groups == 7 and brain.message.blocks == 28
    rows = [mn.run_life({**spec, "module": 64, "batches": 2, "panel_batches": 1,
                         "held_probes": 1, "long_probes": 1})
            for spec in mn.sixth_plan((0,))["sixth"]]
    lines = mn.sixth_verdict(rows, [])
    assert any("-> R1" in line for line in lines) and any("-> R2" in line for line in lines)
    assert any("-> G8" in line for line in lines)


def test_the_verbs_form_is_probed_and_read_again_with_the_message_silent():
    warnings.simplefilter("ignore")
    row = mn.gr.clause(np.random.default_rng(0), 1, agent=(2, 1), v=1)
    assert mn.components(row)["verb_form"] == 3  # the second verb, the plural form
    spec = {"arm": "barred", "module": 64, "settings": mn.BOTH, "founder": 0, "tag": "x",
            "batches": 2, "panel_batches": 1, "probe_scale": 0.0, "held_probes": 1,
            "silent_reading": True}
    life = mn.run_life(spec)
    assert {"said_held_silent", "probes_held_silent", "panels_held_silent"} <= set(life)
    assert "settled" in life["probes_held_silent"]["verb_form"]["verb"]


def test_a_word_is_said_and_taught_by_its_parts():
    warnings.simplefilter("ignore")
    assert all(mn.parts_word(*mn.word_parts(w)) == w for w in range(gr.ACTIONS))
    assert mn.word_parts(9) == (gr.NOUNS, mn.PLURAL)  # "see": the first verb, a plural agent
    brain = mn.make_brain(0, {"parts": True})
    assert type(brain).__name__ == "PartsBrain" and len(brain.motor_index) == sum(mn.PART_SLOTS)
    x = np.tile(gr.concept("GO"), (2, 1))
    brain.lesson(x, np.array([9, 4]))
    said = brain.act(x, greedy=True)
    assert said.shape == (2,) and all(0 <= w < gr.ACTIONS for w in said)
    assert type(copy.deepcopy(brain)).__name__ == "PartsBrain"
    plan = mn.seventh_plan((0, 1))
    assert {s["tag"] for s in plan["seventh"]} == {"words_256", "parts_256", "w10_256",
                                                   "words_free", "parts_free"}
    assert len(plan["seventh_controls"]) == 2 * 2 * 2
    rows = [mn.run_life({**spec, "module": 64, "batches": 2, "panel_batches": 1,
                         "held_probes": 1}) for spec in mn.seventh_plan((0,))["seventh"]]
    lines = mn.seventh_verdict(rows, [], of=1)
    assert any(line.startswith("  P1:") for line in lines)
    assert sum("-> P3" in line for line in lines) == 2


def test_the_sentence_instrument_says_back_and_probes_the_heard_state():
    warnings.simplefilter("ignore")
    row = gr.clause(np.random.default_rng(1), 1)
    back = mn.sayback(row)
    assert back.target == row.target and len(back.pre) == len(row.target)
    assert all(x[w] == 1.0 and x.sum() == 1.0 for x, w in zip(back.pre, row.target, strict=True))
    assert set(mn.sayback_panels(0, 1)) == {"back", "back_held", "back_long"}
    plan = mn.floors_plan((0, 1))
    assert {s["tag"] for s in plan["floors"]} == {"default_64", "plain_256", "slow_256",
                                                  "both_256"}
    rows = [mn.run_life({**spec, "module": 64, "batches": 2, "panel_batches": 1,
                         "held_probes": 1}) for spec in mn.floors_plan((0,))["floors"]]
    by = {r["tag"]: r for r in rows}
    assert set(by["plain_256"]["sayback"]) == {"back", "back_held", "back_long"}
    spot = by["both_256"]["probes_back"]["agent_noun"]["heard"]
    assert {"settled", "message", "joint"} <= set(spot)
    assert "heard" in by["plain_256"]["probes_held"]["verb"]
    control = mn.run_life({"kind": "sentence_control", "model": "lstm", "mode": "online",
                           "founder": 0, "width": 8, "batches": 2, "panel_batches": 1,
                           "held_probes": 1, "parameters": 100})
    assert "heard" in control["probes_back"]["agent_noun"] and "back" in control["panels"]
    lines = mn.floors_verdict(rows, [control])
    assert any(line.startswith("| `both_256`") for line in lines)
    assert any(line.startswith("| lstm, online") for line in lines)


def test_an_echo_names_the_word_of_a_heard_clause_moment():
    row = gr.clause(np.random.default_rng(0), 1, agent=(2, 1), patient=(0, 0), v=1,
                    mods=[(3, 0)], inverted=False)
    named = [int(mn.echo_labels([row], t)[0]) for t in range(len(row.pre))]
    assert named == [-(gr.NOUNS + 2), 6, 3, 0]  # the verb's stem, foxes, owl, dog
    assert mn.echo_labels([mn.sayback(row)], 1).tolist() == [row.target[1]]
    assert mn.echo_labels([gr.path(np.random.default_rng(0), 1)], 0) is None
