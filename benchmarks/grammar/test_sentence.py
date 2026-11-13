"""The sentence instrument's store (cadence#308): the lessons' roles, the store beside the
library's, a life and its reading."""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import grammar as gr  # noqa: E402
import meaning as mn  # noqa: E402
import sentence as st  # noqa: E402


def _names(design, labels):
    return [None if label is None else design.roles[label[2]][0] for label in labels]


def test_a_clauses_moments_name_their_roles_heard_either_way():
    row = gr.clause(np.random.default_rng(0), 2, agent=(2, 1), patient=(0, 0), v=1,
                    mods=[(3, 0), (1, 1)], inverted=False)
    for design in st.DESIGNS.values():
        heard = [st.heard_label(row, t, design) for t in range(len(row.pre))]
        assert _names(design, heard) == ["verb", "agent", "distractor", "distractor", "patient"]
        assert heard[0][:2] == (gr.NOUNS + 1, mn.NO_FORM)  # a verb concept names its stem
        assert heard[1][:2] == mn.word_parts(gr.noun(2, 1))
        said = [st.said_label(row, j, design) for j in range(len(row.target))]
        assert _names(design, said) == ["none", "none", "said_distractor", "none",
                                        "said_distractor", "none", "none", "none"]
        back = mn.sayback(row)
        words = [st.heard_label(back, t, design) for t in range(len(back.pre))]
        assert _names(design, words) == ["agent", "none", "distractor", "none", "distractor",
                                         "verb", "patient", "none"]
        assert [w[:2] for w in words] == [mn.word_parts(w) for w in row.target]


def test_brackets_are_settled_when_heard_or_when_said_by_the_design():
    row = gr.dyck(np.random.default_rng(0), [1, 1, 0, 1], [0, 1, 0, 0])  # ( [ ] ( ) ) end
    context, alone = st.DESIGNS["context"], st.DESIGNS["alone"]
    heard = [st.heard_label(row, t, context) for t in range(4)]
    assert _names(context, heard) == ["open", "open", "close", "open"]
    assert heard[2][:2] == mn.word_parts(gr.CLOSE_SQUARE)  # the bracket the close closes
    said = [st.said_label(row, j, context) for j in range(len(row.target))]
    assert _names(context, said) == ["said_bracket"] * 4 + ["said_close"] * 2 + ["none"]
    heard = [st.heard_label(row, t, alone) for t in range(4)]
    assert _names(alone, heard) == ["bracket"] * 4 and heard[2][0] == st.SILENCE
    said = [st.said_label(row, j, alone) for j in range(len(row.target))]
    assert _names(alone, said) == ["said_open", "said_open", "said_match", "said_open",
                                   "said_close", "said_close", "none"]
    path = gr.path(np.random.default_rng(0), 2)
    labels = [st.heard_label(path, t, alone) for t in range(len(path.pre))]
    assert labels[:-1] == [None] * (len(path.pre) - 1)  # the facts are heard without a lesson
    assert _names(alone, labels[-1:]) == ["start"]
    both = st.DESIGNS["alone_path"]  # the facts echoed, the start leaving once said
    labels = [st.heard_label(path, t, both) for t in range(len(path.pre))]
    assert _names(both, labels) == ["none"] * (len(path.pre) - 1) + ["start"]
    assert [label[:2] for label in labels[:4]] == [
        mn.word_parts(int(np.argmax(path.pre[t][:gr.ACTIONS]))) for t in range(4)]
    said = [st.said_label(path, j, both) for j in range(len(path.target))]
    assert _names(both, said) == ["said_start"] + ["none"] * (len(path.target) - 1)
    assert both.base == alone.base + 1 and both.inventory(False)[1][2] == both.base


def test_two_wordings_and_questions_name_what_is_asked():
    row = gr.clause(np.random.default_rng(0), 1, agent=(2, 1), patient=(0, 0), v=1,
                    mods=[(3, 0)], inverted=False)
    alone = st.DESIGNS["alone"]
    second = st.wording(row, False, True)  # the first grammar's words, the second said
    assert second.target == gr.twin(row, dialect=True).target
    heard = [st.heard_label(second, t, alone) for t in range(len(second.pre))]
    assert _names(alone, heard) == ["agent", "none", "distractor", "verb", "patient", "none",
                                    "dialect"]
    first = st.wording(row, True, False)  # the second grammar's words, the first said
    heard = [st.heard_label(first, t, alone) for t in range(len(first.pre))]
    assert _names(alone, heard) == ["agent", "none", "distractor", "patient", "verb", "none"]
    assert first.target == row.target
    concepts = gr.twin(row, dialect=True)
    assert _names(alone, [st.heard_label(concepts, 0, alone)]) == ["verb_dialect"]
    for ask, answer in (("agent", gr.noun(2, 1)), ("patient", gr.noun(0, 0)),
                        ("verb", gr.verb(1, 1))):
        for words in (False, True):
            asked = st.question(row, ask, words)
            assert asked.target == [answer, gr.END]
            last = st.heard_label(asked, len(asked.pre) - 1, alone)
            assert last == (st.SILENCE, mn.NO_FORM, alone.op["question"])
            assert _names(alone, [st.said_label(asked, j, alone) for j in range(2)]) == [
                "none", "none"]
    rng = np.random.default_rng(1)
    split = gr.Split.of(0)
    for _ in range(20):
        batch = st.full_batch(rng, split)
        assert len({len(r.pre) for r in batch}) == 1 and len({len(r.target) for r in batch}) == 1
    panels = st.more_panels(0, 1)
    assert set(panels) == set(st.MORE)
    held = [r for rows in panels["para_12_held"] for r in rows]
    assert all(split.held(tuple(r.meta["agent"]), tuple(r.meta["patient"]), r.meta["verb"])
               for r in held)


def test_the_instruments_store_is_the_librarys_to_the_bit():
    warnings.simplefilter("ignore")
    for design, train in (("context", "sayback"), ("alone", "all")):
        rows = [st.life({"kind": "store", "founder": 0, "batches": 6, "module": 32,
                         "panel_batches": 1, "held_probes": 1, "design": design,
                         "train": train, "emulated": emulated, "tag": "small"})
                for emulated in (True, False)]
        for key in ("parameters", "synapses", "panels", "sayback", "kinds", "probes_back",
                    "gains"):
            assert json.dumps(rows[0][key], sort_keys=True) == json.dumps(
                rows[1][key], sort_keys=True), (design, key)


def test_a_life_reads_its_gates_beside_its_controls():
    warnings.simplefilter("ignore")
    spec = {**st.STORE, "founder": 0, "batches": 4, "module": 32, "panel_batches": 1,
            "held_probes": 1, "checkpoints": [2], "tag": "store"}
    row = st.run_life(spec)
    assert row["full"] and row["design"] == "alone" and set(row["more"]) == set(st.MORE)
    assert row["curve"][0]["batches"] == 2 and "ask_words_held" in row["curve"][0]
    assert {"g1_to_g2", "g2_to_g1"} == set(row["probes_cross"])
    assert "joint" in row["probes_back"]["agent_noun"]["heard"]
    controls = [st.run_life({"kind": "store_control", "model": model, "mode": "offline",
                             "founder": 0, "width": 8, "batches": 4, "panel_batches": 1,
                             "held_probes": 1, "parameters": row["parameters"],
                             "train": "all", **roles})
                for model in ("lstm", "transformer") for roles in ({}, {"roles": "alone"})]
    assert [c["model"] for c in controls] == ["lstm", "lstm with roles", "transformer",
                                              "transformer with roles"]
    assert all(set(st.MORE) <= set(c["panels"]) for c in controls)
    lines = st.read([row], controls, of=1)
    assert any("H1" in line and "H2" in line for line in lines)
    assert any("H3 said" in line and "H4" in line for line in lines)
    assert any("less lstm with roles" in line for line in lines)
    plan = st.second_plan((0, 1))
    assert len(plan["second"]) == 2 and len(plan["second_controls"]) == 2 * 2 * 3


def test_a_quiet_echo_leaves_the_reads_address_as_it_is():
    warnings.simplefilter("ignore")
    design = st.DESIGNS["alone"]
    brain = st.make_brain(0, {"learning_surprise": False}, 32, st.DEPTH, True, design)
    recall = brain.recall
    rates = {name: getattr(recall, name) for name in st._READ_RATES}
    row = gr.clause(np.random.default_rng(0), 1)
    x = row.pre[0][None, :]
    labels = np.array([st.heard_label(row, 0, design)])
    brain.act(x, greedy=True)
    before = [np.array(a, copy=True) for a in recall.head_map(0)]
    work = gr.new_work()
    assert st._lesson(brain, x, labels, work, said=False, quiet=True)
    assert all(np.array_equal(a, b) for a, b in zip(before, recall.head_map(0), strict=True))
    assert {name: getattr(recall, name) for name in st._READ_RATES} == rates
    assert st._lesson(brain, x, labels, work, said=False)
    assert any(not np.array_equal(a, b) for a, b in zip(before, recall.head_map(0), strict=True))


def test_a_paths_start_can_clear_the_store():
    warnings.simplefilter("ignore")
    clean = st.DESIGNS["alone_clean"]
    start = dict(clean.roles)["start"]
    assert start[0] == ("set", st.START) and {g for _, g in start[1:]} == (
        set(range(len(st.GROUPS))) - {st.START})
    assert clean.inventory(False)[0] == len(st.GROUPS)
    brain = st.make_brain(0, {"learning_surprise": False}, 16, st.DEPTH, False, clean)
    store = st.Store(16, clean.roles, st.DEPTH, len(st.GROUPS))
    store.reset(1)
    store.apply(np.array([clean.op["agent"]]), np.ones((1, 16)))
    store.apply(np.array([clean.op["start"]]), np.full((1, 16), 2.0))
    heads = store.heads().reshape(len(st.GROUPS), 16)
    assert not heads[st.AGENT].any() and (heads[st.START] == 2.0).all()
    assert brain.message.groups == len(st.GROUPS)
