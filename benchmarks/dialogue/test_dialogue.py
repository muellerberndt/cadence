"""Contracts of the conversation chamber (cadence#305): the synthetic world, the dataset
adapters, the brain's lives and the matched controls."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import chamber as ch  # noqa: E402
import world as wd  # noqa: E402


def test_the_world_s_conversations_align_and_keep_held_pairs_out_of_lessons():
    world = wd.World(0)
    assert world.lexicon.inputs == 31 and world.lexicon.actions == 8
    rng = np.random.default_rng(1)
    for wording in ("trained", "held", "prefix"):
        batch = world.conversations(rng, 5, 6, 2, lesson=wording == "trained", wording=wording)
        for s in range(6):
            assert len({len(c[s]) for c in batch}) == 1  # one shape: turns align word for word
            for t in range(len(batch[0][s])):
                assert len({len(world.lexicon.moments(c[s][t])) for c in batch}) == 1
        lesson = wording == "trained"
        for conversation in batch:
            for session in conversation:
                for turn in session:
                    if not turn.asked:
                        assert turn.said[turn.turn_at] == turn.tags["place"]
                        if lesson:
                            assert (turn.tags["about"], turn.tags["place"]) not in world.held
                    elif wording == "prefix":
                        assert turn.answer not in turn.said
    gaps = {t.tags["gap"] for c in batch for s in c for t in s if t.asked}
    assert 0 in gaps and max(gaps) >= 2


def test_a_question_s_moments_end_with_the_turn_and_a_silent_answer():
    world = wd.World(0)
    told = world.tell("mary", "garden", 0)
    asked = world.ask("mary", "garden", "trained", 0, told)
    lex = world.lexicon
    tell_moments = lex.moments(told)
    assert [int(x.argmax()) for x, _ in tell_moments][4] == lex.index[wd.TURN]
    assert all(label is None for _, label in tell_moments)
    moments = lex.moments(asked)
    assert moments[-2][0][lex.index[wd.TURN]] == 1.0 and not moments[-1][0].any()
    assert moments[-1][1] == lex.action["garden"]
    held = world.ask("mary", "garden", "held", 1, told)
    assert set(held.said) <= set(lex.words) and held.said != asked.said
    assert world.ask("mary", "garden", "prefix", 0, told).said == ["mary", "went", "to", "the"]


def test_the_babi_and_dialog_adapters_read_their_formats(tmp_path):
    qa = tmp_path / "qa1_single-supporting-fact_train.txt"
    qa.write_text("1 Mary moved to the bathroom.\n2 John went to the hallway.\n"
                  "3 Where is Mary? \tbathroom\t1\n1 Sandra travelled to the office.\n"
                  "2 Where is Sandra? \toffice\t1\n")
    stories = wd.babi_qa(qa)
    assert len(stories) == 2 and len(stories[0][0]) == 3
    first = stories[0][0][0]
    assert first.said == ["mary", "moved", "to", "the", "bathroom", "."] and first.turn_at == 4
    assert stories[0][0][2].answer == "bathroom" and stories[0][0][2].tags["support"] == [1]
    lexicon = wd.babi_lexicon(stories)
    assert lexicon.answers == ["bathroom", "office"] and lexicon.words[-1] == wd.TURN
    train, test = tmp_path / "trn.txt", tmp_path / "tst.txt"
    train.write_text("1 hi\thello what can i help you with today\n2 thanks\tyou are welcome\n")
    test.write_text("1 hi\thello what can i help you with today\n"
                    "2 resto_a R_phone resto_a_phone\n3 phone please\there it is resto_a_phone\n")
    assert wd.dialog_turns(test)[0][1] == ("resto_a R_phone resto_a_phone", "")
    answers = wd.dialog_answers(train, test)
    assert answers == {"turns": 2.0, "whole": 0.5, "words": 0.5}


@pytest.mark.parametrize("arm", ch.ARMS)
def test_every_arm_lives_and_scores_its_answers(arm):
    rows = ch.synthetic_cell({"founder": 0, "arm": arm, "rounds": 2, "sessions": 3,
                              "readings": [{"batches": 1}, {"batches": 1, "erase": "records"},
                                           {"batches": 1, "erase": "message"}]})
    assert len(rows) == 3
    for row in rows:
        assert row["scores"]["all"]["questions"] == ch.STREAMS * 5
        assert 0 <= row["scores"]["all"]["right"] <= 1 and row["answer_ms"] > 0
    brain = ch.make_brain(wd.World(0).lexicon, 0, arm)
    assert (getattr(brain, "longterm", None) is not None) == (arm.startswith("records")
                                                              or arm == "turns")
    assert (getattr(brain, "message", None) is not None) == (arm != "previous_word")
    assert rows[0]["parameters"] == brain.parameters()


def test_the_records_arm_files_a_record_under_each_fact_s_words_before_its_turn():
    world = wd.World(0)
    brain = ch.make_brain(world.lexicon, 0, "records")
    batch = world.conversations(np.random.default_rng(3), 2, 3, 2, lesson=False,
                                wording="prefix")
    ch.live(brain, world.lexicon, batch, "records", teach=False)
    for stream, conversation in enumerate(batch):
        keys = {tuple(t.said[: t.turn_at]) for s in conversation for t in s if not t.asked}
        # one record per distinct wording before the turn: a fact told again in its own words
        # revises its record, in other words it takes another
        assert brain.longterm.stores[stream].count == len(keys)


def test_the_plans_parse():
    plans = {p.stem: json.loads(p.read_text()) for p in (HERE / "step0").glob("*.json")}
    assert set(plans) == {"synthetic", "babi", "controls-synthetic", "controls-babi", "keys"}
    todo = ch.cells(plans["babi"])
    assert len(todo) == 3 * 3 * 4 and {c["task"] for c in todo} == {1, 2, 6}
    assert len(ch.cells(plans["synthetic"])) == 12


def test_the_controls_match_the_brain_and_read_the_same_questions():
    pytest.importorskip("torch")
    import controls as co

    rows = co.run_cell({"founder": 0, "model": "lstm", "mode": "online", "rounds": 2,
                        "sessions": 3, "readings": [{"batches": 1}]})
    brain = ch.make_brain(wd.World(0).lexicon, 0, "records")
    assert rows[0]["target"] == brain.parameters()
    assert abs(rows[0]["parameters"] - brain.parameters()) / brain.parameters() < 0.02
    assert rows[0]["scores"]["all"]["questions"] == ch.STREAMS * 5
    lexicon = wd.World(0).lexicon
    x, y, where = co.encode(lexicon, wd.World(0).conversations(np.random.default_rng(0), 2, 2,
                                                               2, lesson=True))
    assert x.shape[0] == 2 and len(where) == 2 * 3 and (y >= 0).sum() == len(where)


def test_the_key_reading_finds_the_prefix_whole_and_the_questions_apart():
    rows = {r["wording"]: r for r in ch.keys_cell({"founder": 0})}
    assert rows["prefix"]["cosine"] == 1.0 and rows["prefix"]["at_match"] == 1.0
    assert rows["trained"]["at_match"] == 0.0 and rows["trained"]["cosine"] < 0.5


def test_the_declared_plans_parse_on_fresh_founders():
    plans = {p.stem: json.loads(p.read_text()) for p in (HERE / "declared").glob("*.json")}
    assert set(plans) == {"synthetic", "babi", "controls-synthetic", "controls-babi"}
    for plan in plans.values():
        assert plan["founders"] == list(range(3050, 3056))
    todo = ch.cells(plans["synthetic"])
    assert len(todo) == 18 and {c.get("label", c["arm"]) for c in todo} == {
        "turns", "turns_nearest", "previous_word"}
    names = [r["name"] for r in plans["synthetic"]["readings"]]
    assert names == ["R1", "R2", "R3", "R4", "R5", "R6"]
    assert len(ch.cells(plans["babi"])) == 2 * 6 * 3


def test_a_world_with_a_gap_asks_exactly_that_far_back_at_every_anchor():
    world = wd.World(0, people=24)
    batch = world.conversations(np.random.default_rng(0), 4, 21, 2, lesson=False, gap=10)
    anchored = [t for c in batch for s in c for t in s if t.tags.get("anchor")]
    assert len(anchored) == 4 * 2 and {t.tags["gap"] for t in anchored} == {10}


def test_the_turns_arm_answers_from_its_records_across_sessions():
    rows = ch.synthetic_cell({"founder": 0, "arm": "turns", "people": 24, "rounds": 60,
                              "sessions": 4, "readings": [
                                  {"batches": 1, "sessions": 11, "gap": 10},
                                  {"batches": 1, "sessions": 11, "gap": 10, "erase": "records"}]})
    assert rows[0]["scores"]["anchored"]["right"] >= 0.9
    assert rows[1]["scores"]["anchored"]["right"] <= 0.5
    assert rows[0]["answer_macs"] > 0


def test_the_declared_receipts_hold_the_verdicts_the_readme_states():
    import summarize as sm

    found = []
    for path in sorted((HERE / "results" / "declared").glob("*.jsonl.gz")):
        found += sm.rows(path.with_name(path.name.removesuffix(".gz")))
    assert len(found) == 108 + 72 + 96 + 48
    gates = sm.gates(found)
    assert all(gate["passes"] for gate in gates.values())
    assert gates["T1"]["values"] == [1.0] * 6 and gates["T4"]["founders"] == 5
    readme = (HERE / "README.md").read_text()
    assert "| T4 held and used | R5, records emptied, ten sessions back |" in readme
    babi = [r for r in found if r.get("arm") == "turns" and r["chamber"] == "babi-1"]
    plain = {r["founder"]: r["scores"]["all"]["right"] for r in babi
             if not r["reading"].get("erase")}
    for row in babi:
        if row["reading"].get("erase"):
            assert abs(row["scores"]["all"]["right"] - plain[row["founder"]]) <= 0.001 + 1e-9
