"""The night's chamber (cadence#334): a tiny life runs, resumes, and lives with a control in the
brain's place; the page's host steps a saved brain."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import night as nt  # noqa: E402

TINY = {k: 1 for k in ("N0", "N0b", "N1", "N2", "N3", "N4", "N5", "N6")}


def test_every_question_form_has_one_right_answer_in_its_room():
    rng = np.random.default_rng(0)
    held = nt.founder_held(0)
    for form in ("name", "colour_of", "by_colour", "round", "where", "left_of", "right_of",
                 "next_to", "opposite", "material", "eat", "can_eat", "can_eat_this"):
        ex = nt.exchange(nt.Parrot(0, streams=1), rng, form, "turn", held)
        assert ex.question is not None and ex.question.answer
        assert ex.turn == len(ex.heard) - 1


def test_a_tiny_life_reads_every_step_and_resumes_from_its_checkpoint(tmp_path):
    first = nt.life(3, blocks=TINY, streams=2, reads=2, master=False, ckpt=str(tmp_path),
                    steps=["N0", "N1"])
    assert sorted(first["steps"]) == ["N0", "N1"]
    meta = json.loads((tmp_path / "ckpt_3.json").read_text())
    assert meta["step"] == 3 and meta["block"] == 0  # N0, N0b, N1 lie behind it
    again = nt.life(3, blocks=TINY, streams=2, reads=2, master=False, ckpt=str(tmp_path),
                    steps=["N0", "N1", "N2"])
    assert sorted(again["steps"]) == ["N0", "N1", "N2"]


def test_a_control_lives_the_same_exchanges_in_the_brains_place():
    got = nt.life(4, blocks=TINY, streams=2, reads=2, master=False, steps=["N0", "N1"],
                  control="table")
    assert set(got["steps"]["N1"]) >= {"imitate", "name"}


def test_the_page_host_steps_a_saved_brain_and_takes_a_pistachio(tmp_path):
    from cadence.meaning.host import MeaningParrot

    p = nt.Parrot(5, streams=1)
    nt.export_spec(p, tmp_path / "spec.json")
    p.brain.save(tmp_path / "brain.npz")
    m = MeaningParrot(tmp_path / "brain.npz", tmp_path / "spec.json")
    m.see([{"kind": "apple", "colour": "red", "material": "fruit", "direction": 1.0,
            "distance": 1.4, "focus": True}])
    for word in ("what", "is"):
        m.hear(word)
        assert m.moment(more=True) is None  # held while the trainer has more to say
    m.hear("this")
    said = [m.moment(), m.moment(), m.moment()]
    assert all(w is None or w in nt.WORDS for w in said)
    assert m.pistachio() == sum(w is not None for w in said)


def test_the_word_records_are_kept_with_the_checkpoint_and_the_brain(tmp_path):
    nt.life(6, {"word_records": 1}, blocks=TINY, streams=2, reads=2, master=False,
            ckpt=str(tmp_path / "ck"), save=str(tmp_path / "brains"), steps=["N0", "N1"])
    kept = tmp_path / "ck" / "ckpt_6.records.npz"  # a correction's answer is kept as a record
    assert kept.exists() and (tmp_path / "brains" / "brain_6_N1.records.npz").exists()
    p = nt.Parrot(6, {"word_records": 1}, streams=2)
    assert nt.load_records(p, kept) and p.records.count > 0


@pytest.mark.parametrize("key", ["association", "input"])
def test_the_page_host_keeps_a_correction_and_recalls_it_at_the_turn(tmp_path, key):
    from cadence.meaning.host import MeaningParrot

    p = nt.Parrot(7, {"word_records": 1, "records_key": key}, streams=1)
    nt.export_spec(p, tmp_path / "spec.json")
    p.brain.save(tmp_path / "brain.npz")
    nt.save_records(p, tmp_path / "brain.records.npz")
    m = MeaningParrot(tmp_path / "brain.npz", tmp_path / "spec.json")
    assert m.records is not None and m.records.count == 0
    apple = {"kind": "apple", "colour": "red", "material": "fruit", "direction": 1.0,
             "distance": 1.4, "focus": True}
    m.see([apple])
    for word in ("what", "is", "this"):
        m.hear(word)
        m.moment(more=True)
    assert m.correct("apple") and m.records.count == 1
    assert np.array_equal(m.recall(m.cue), m.code("apple"))  # the same pattern recalls it
    m.moment()  # the turn opens in the gap: the record is replayed to the ear
    assert np.array_equal(m.recalled, m.code("apple"))


def test_a_nights_sleep_scales_every_efficacy_down():
    p = nt.Parrot(8, streams=1)
    before = np.asarray(p.brain.learner.brain.efficacy).copy()
    nt.sleep(p, 0.5)
    assert np.allclose(np.asarray(p.brain.learner.brain.efficacy), before * 0.5)


@pytest.mark.parametrize("balance", [0, 1])
def test_the_page_host_reads_the_mixed_code_as_the_chamber_does(tmp_path, balance):
    from cadence.meaning.host import MeaningParrot

    p = nt.Parrot(9, {"mixed_code": 64, "mixed_balance": balance}, streams=1)
    nt.export_spec(p, tmp_path / "spec.json")
    p.brain.save(tmp_path / "brain.npz")
    m = MeaningParrot(tmp_path / "brain.npz", tmp_path / "spec.json")
    apple = {"kind": "apple", "colour": "red", "material": "fruit", "direction": 1.0,
             "distance": 1.4, "focus": True}
    m.see([apple])
    m.hear("what")
    m.moment(more=True)
    x = m.observation()
    assert x.shape == (1, p.inputs)
    code = x[0, p.inputs - p.ear - p.mixed: p.inputs - p.ear]
    assert code.sum() == p.mixed_on  # the sparse code: a tenth of its units on
    trace = m.trace.copy()  # the chamber's code of the same glimpse and trace
    assert np.array_equal(code, p.mixed_code(x[0, : p.eyes.slot_width], trace))
