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
        "distractor2_noun": 3, "distractor2_number": 1}
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
