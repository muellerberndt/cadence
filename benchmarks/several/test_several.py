"""Contracts of the several-records chamber (cadence#325): the world's facts and questions, one
small life of the brain's arms, the controls' window, and the summary."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import several_chamber as ch  # noqa: E402
import several_summary as sm  # noqa: E402
import several_world as sw  # noqa: E402

SMALL = {"founder": 0, "module": 32, "lesson_subjects": 60, "lessons": 20, "sizes": [150],
         "questions": 2}


def test_a_word_is_heard_as_its_form_and_a_fact_marks_its_turn():
    lexicon = sw.Lexicon()
    assert lexicon.inputs == sw.WIDTH + 1 and lexicon.actions == sw.SLOTS * (len(sw.PLACES) + 1)
    x = lexicon.moment("e7")
    assert x.sum() == sw.ON and not x[sw.WIDTH] and np.array_equal(x, lexicon.moment("e7"))
    assert not np.array_equal(x, lexicon.moment("e8")) and not lexicon.moment(None).any()
    told = lexicon.told(sw.Fact("e7", "home", "place03"))
    assert len(told) == 8 and told[5][sw.WIDTH] == 1.0  # the turn before the object
    assert np.array_equal(told[6], lexicon.moment("place03"))


def test_a_population_tells_each_fact_once_and_its_questions_name_their_records():
    people = sw.Population([0, 325, 9], 200, "e")
    assert len({(f.relation, f.subject) for f in people.facts}) == len(people.facts)
    assert all(people.where[(f.relation, f.subject)] == i for i, f in enumerate(people.facts))
    rng = np.random.default_rng(0)
    lexicon = sw.Lexicon()
    for k in (1, 2, 4, 8):
        q = people.parts(rng, k)
        assert q.kind == "parts" and q.k == k and len(q.needs) == k and len(q.answer) == sw.SLOTS
        assert q.answer[k:] == [sw.NOTHING] * (sw.SLOTS - k)
        for j, i in enumerate(q.needs):
            fact = people.facts[i]
            assert fact.answer == q.answer[j] and fact.relation == q.said[3]
            assert fact.subject == q.said[5 + 2 * j]
        moments, labels = lexicon.asked(q)
        assert len(moments) == len(q.said) + 2 and labels.shape == (sw.SLOTS,)
        assert moments[-2][sw.WIDTH] == 1.0 and not moments[-1].any()
    for hops in (2, 3):
        q = people.chain(rng, hops)
        assert q.kind == "chain" and len(q.needs) == hops and q.answer[1:] == [sw.NOTHING] * 7
        at = q.said[-2]
        for i in q.needs:  # each hop's record is about what the hop before it said
            assert people.facts[i].subject == at
            at = people.facts[i].answer
        assert at == q.answer[0] and at in sw.PLACES
    lost = people.parts(rng, 4, absent=1.0)
    assert lost.needs == [-1] * 4 and lost.answer == [sw.NOTHING] * sw.SLOTS
    assert all((lost.said[3], s) not in people.where for s in lost.said[5::2])
    broken = people.chain(rng, 3, broken=1.0)
    assert broken.answer == [sw.NOTHING] * sw.SLOTS and broken.needs.count(-1) >= 1
    kept = people.keep(100)
    assert len(kept.facts) == 100 and all(i < 100 for i in kept.where.values())


def test_a_small_life_of_the_learned_heads_reads_and_scores_every_group():
    rows = ch.run_cell({**SMALL, "arm": "attention",
                        "longterm": {"attention": 1.0, "motor": 1.2}})
    (row,) = rows
    assert row["arm"] == "attention" and row["size"] == 150 and row["records"] == 150
    assert row["heads"] == sw.SLOTS and len(row["nursery_curve"]) == 10
    scores = row["scores"]
    for group in ("parts 1", "parts 2", "parts 4", "parts 8", "chain 2", "chain 3"):
        for kind in ("said", "part", "delivered", "ordered"):
            # two questions and one with parts absent at .3 (or a broken chain): three in all
            held = [scores[name]["questions"] for name in (f"{kind} {group}",
                                                           f"{kind} {group} absent")
                    if name in scores]
            assert sum(held) == 3
        assert row["costs"][group]["read_macs"] > 0 and row["costs"][group]["read_ms"] > 0
    # a newborn's heads take the question's subjects in the order heard: parts lie in order
    assert scores["ordered parts 8"]["right"] == 1.0 and scores["delivered parts 4"]["right"] == 1.0
    assert scores["refused chain 2 absent"]["questions"] == 1


def test_the_oracle_hands_every_head_its_record_and_the_plain_brain_holds_none():
    (row,) = ch.run_cell({**SMALL, "arm": "oracle", "longterm": {"rest": True}, "oracle": True})
    for group in ("parts 1", "parts 8", "chain 3"):
        assert row["scores"][f"ordered {group}"]["right"] == 1.0
    (plain,) = ch.run_cell({**SMALL, "arm": "none", "records": False})
    assert plain["records"] is None and "delivered parts 1" not in plain["scores"]
    assert plain["parameters"] < row["parameters"]


def test_the_controls_window_holds_what_a_question_needs():
    pytest.importorskip("torch")
    import several_controls as ct

    lexicon = sw.Lexicon()
    people = sw.Population([0, 325, 9], 200, "e")
    rng = np.random.default_rng(1)
    for question in (people.parts(rng, 8), people.chain(rng, 3), people.parts(rng, 2, 1.0)):
        facts = ct.context(people, question, rng, 64)
        assert len(facts) == 64 and len(set(facts)) == 64
        assert all(people.facts[i] in facts for i in question.needs if i >= 0)
        moments, labels = ct.sequence(lexicon, people, question, rng, 64)
        assert moments.shape == (64 * 8 + len(question.said) + 2, lexicon.inputs)
    width = ct.matched_width("transformer", 60_000, lexicon.inputs, lexicon.slot_actions, 2, 2)
    net = ct.build("transformer", width, lexicon.inputs, lexicon.slot_actions, 0, 2, 2)
    assert abs(ct.count(net) - 60_000) < 6_000
    rows = ct.run_cell({"founder": 0, "kind": "lstm", "schedule": "online", "width": 16,
                        "lessons": 32, "lesson_subjects": 60, "sizes": [150], "questions": 2,
                        "window": 64, "layers": 1, "attention_heads": 2})
    assert rows[0]["arm"] == "lstm, online" and rows[0]["scores"]["said parts 8"]["questions"] == 2


def test_the_summary_reads_ranges_and_bars():
    assert sm.spread([0.5, 0.5]) == ".500" and sm.spread([0.25, 1.0]) == ".250 to 1.000"
    assert sm.at_bar([0.96, 0.94, 1.0], 0.95) == "2 of 3"
    rows = [{"arm": "a", "size": 10, "parameters": 5, "founder": f,
             "scores": {"said parts 1": {"right": 0.5 + f / 4, "questions": 4}},
             "costs": {"parts 1": {"answer_ms": 1.0, "read_ms": 0.1, "answer_macs": 10,
                                   "read_macs": 2}}} for f in (0, 1)]
    lines = sm.table(rows, ("said",))
    assert "| `a` | 10 | 2 | 5 | said | .500 to .750 |" in lines[2]
    assert any("read_ms" in line for line in sm.costs(rows))


DECLARED, ONLINE = tuple(range(3250, 3262)), ("transformer", "lstm")  # PROTOCOL.md


def _row(founder: int, arm: str = "heads", right: float = 1.0) -> dict:
    """A row at 100,000 records holding every reading the gates name, each at ``right``."""
    scores = {f"{kind} {g}": {"right": right, "questions": 200}
              for kind in ("delivered", "said") for g in sm.GROUPS}
    scores |= {f"silent parts {k} absent": {"right": right, "questions": 100} for k in (2, 4, 8)}
    scores |= {f"refused {g} absent": {"right": right, "questions": 100}
               for g in ("parts 1", "chain 2", "chain 3")}
    if arm == "birth":
        scores["said chain 3"]["right"] = 0.0
    row = {"arm": arm, "size": 100_000, "founder": founder, "parameters": 1, "scores": scores,
           "costs": {f"parts {k}": {"read_ms": 1.0, "answer_ms": 1.0, "answer_macs": 1,
                                    "read_macs": 1} for k in (1, 2, 4, 8)}}
    if arm in ONLINE:
        row |= {"arm": f"{arm}, online", "kind": arm, "schedule": "online"}
        row["scores"] = {name: dict(score, right=0.0) for name, score in scores.items()}
    return row


def _panel(founders=DECLARED, controlled=DECLARED[:6], kinds=ONLINE) -> list[dict]:
    return ([_row(f, arm) for f in founders for arm in ("heads", "birth")]
            + [_row(f, kind) for f in controlled for kind in kinds])


def _verdicts(rows: list[dict]) -> list[str]:
    return [line.rsplit("|", 2)[1].strip() for line in sm.gates(rows) if line.startswith("| A")]


def test_the_gates_read_incomplete_until_the_declared_panel_is_given_and_name_it():
    assert set(_verdicts(_panel())) == {"passes"}
    for rows in (_panel(founders=[3250], controlled=[3250]),  # one founder of twelve
                 _panel(founders=range(10), controlled=range(6))):  # outside the panel
        assert set(_verdicts(rows)) == {"incomplete"}
        assert sm.gates(rows)[-1].startswith(
            "incomplete: the gates are declared on founders 3250 to 3261")
    nine = _panel(founders=range(3250, 3259))  # nine founders, the bar is ten
    verdicts = _verdicts(nine)
    assert verdicts[:11] + verdicts[16:] == ["incomplete"] * 14
    assert verdicts[11:16] == ["passes"] * 5  # A5's own panel, founders 3250 to 3255, is given
    assert sm.gates(nine)[-1].endswith(
        "not given: `heads` 3259, 3260, 3261; `birth` 3259, 3260, 3261")
    rows = [*_panel(), *_panel(founders=[0], controlled=[0])]
    lines = sm.gates(rows)
    assert set(_verdicts(rows)) == {"passes"}
    assert lines[-1] == "not counted, outside the declared founders: 0"
    assert "| A1 | delivered, parts 2 | 0.95 | 12 of 12 | 1.000 | passes |" in lines


def test_the_bar_of_ten_founders_is_not_lowered_to_the_founders_that_hold_a_reading():
    rows = _panel()
    for row in rows[:6:2]:  # three founders' heads rows without the reading of parts of 2
        del row["scores"]["delivered parts 2"]
    assert "| A1 | delivered, parts 2 | 0.95 | 9 of 9 | 1.000 | fails |" in sm.gates(rows)


def test_a5_waits_for_every_online_control_on_its_six_founders():
    verdicts = _verdicts(_panel(kinds=("transformer",)))
    assert verdicts[10] == "passes" and verdicts[11:16] == ["incomplete"] * 5
    assert sm.gates(_panel(kinds=("transformer",)))[-1].endswith(
        "not given: `lstm, online` 3250, 3251, 3252, 3253, 3254, 3255")


def test_a_second_row_of_a_founders_arm_at_a_size_is_refused():
    for twice in (_row(3250), _row(3250, "birth"), _row(3253, "lstm")):
        with pytest.raises(SystemExit, match="founder 325[03] already has a row of the arm"):
            sm.gates([*_panel(), twice])
    with pytest.raises(SystemExit, match="founder 3250 already has a row of the arm 'heads'"):
        sm.gates([_row(3250)] * 12)
