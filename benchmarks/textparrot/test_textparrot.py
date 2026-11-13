"""The text parrot chamber's contracts."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import textchamber as tc  # noqa: E402
import textparrot as tp  # noqa: E402
import textworld as tw  # noqa: E402


class Silent:
    """A learner that says nothing and holds nothing."""

    def begin(self, n: int) -> None:
        pass

    def moment(self, x, labels, mask, learn):  # type: ignore[no-untyped-def]
        return np.tile([len(tw.VOCABULARY), tc.PASS], (len(x), 1))


@pytest.mark.parametrize("form", tp.ALL_FORMS)
def test_teacher_forcing_says_the_teachers_reply(form: str) -> None:
    rng = np.random.default_rng(1)
    layout = tc.Layout.founder(1)
    items = [tp.draw(form, rng) for _ in range(40)]
    for e in tc.run(Silent(), layout, items, "teacher"):
        got = tc.score(e)
        assert got["right"] == 1.0, got
        if "parsed" in got:  # the parser reads a sentence back to the scene
            assert got["parsed"] == 1.0, got


def test_a_silent_parrot_answers_nothing_right() -> None:
    rng = np.random.default_rng(2)
    layout = tc.Layout.founder(2)
    items = [tp.draw("colour", rng) for _ in range(8)]
    assert all(tc.score(e)["right"] == 0.0 for e in tc.run(Silent(), layout, items, "free"))


def test_held_out_splits_never_reach_pretraining() -> None:
    rng = np.random.default_rng(3)
    for _ in range(400):
        form = str(rng.choice(tp.ALL_FORMS))
        sc, q = tp.draw(form, rng)
        assert not any((t.kind, t.colour) in tw.HELD_PAIRS for t in sc.things)
        assert not any(t.kind in tw.NEW_KINDS for t in sc.things)
        assert (form, q.depth) not in tp.HELD_DEPTH or q.referent is None
        if form in ("relation", "describe_rel"):
            assert (q.meta["relation"], q.referent.kind) not in tw.HELD_RELATIONS
    sc, q = tp.draw("colour", rng, split="pairs")
    assert (q.referent.kind, q.referent.colour) in tw.HELD_PAIRS
    sc, q = tp.draw("relation", rng, split="relations")
    assert (q.meta["relation"], q.referent.kind) in tw.HELD_RELATIONS


def test_the_owners_questions_in_the_owners_room() -> None:
    import textintrospect as ti

    room = ti.owner_room()
    want = {" ".join(w): " ".join(ti.teacher_reply(room, f, w)) for f, w in ti.OWNER}
    assert want["what colour has the apple ?"] == "red ."
    assert want["what is left of the apple ?"] == "block ."
    assert want["where is the cup ?"] == "behind ."
    assert want["what objects are round ?"] == "apple cup ."
    rel = want["what do you see left of the apple ?"]
    assert rel == "the blue block is left of the red apple ."


def test_the_parser_reads_its_grammar_and_refuses_the_rest() -> None:
    p = tw.parse("the red apple is left far the blue cup is behind .".split())
    assert p["bad"] == 0 and p["things"] == [("red", "apple", "left", True),
                                             ("blue", "cup", "behind", False)]
    r = tw.parse("the red apple is in front of the blue cup .".split(), relational=True)
    assert r["bad"] == 0 and r["relations"] == [("red", "apple", "in front of", "blue", "cup")]
    assert tw.parse("red apple left .".split())["bad"] > 0
    assert tw.parse("the red apple is left".split())["bad"] > 0  # no "."


def test_the_eyes_report_no_label_and_the_forms_are_drawn_per_founder() -> None:
    rng = np.random.default_rng(4)
    a = tw.look("apple", "red", ("left", "far"), rng)
    b = tw.look("apple", "red", ("left", "far"), rng)
    assert a.shape == (tw.GLIMPSE,) and not np.array_equal(a, b)  # noisy senses
    assert not np.array_equal(tc.Layout.founder(1).forms, tc.Layout.founder(2).forms)


def test_a_composed_brain_lives_an_exchange() -> None:
    rng = np.random.default_rng(5)
    layout = tc.Layout.founder(5)
    policy = tp.BrainPolicy(layout, {"modules": (16, 16, 16), "apical": 0.25, "message": 12,
                                     "ear_entry": "association", "sleep_scale": 0.5}, 5)
    eps = tc.run(policy, layout, [tp.draw("where", rng) for _ in range(2)], "dagger")
    assert all(e.done for e in eps) and policy.lessons > 0
    before = float(np.abs(np.asarray(policy.brain.learner.brain.efficacy)).max())
    policy.sleep(0.5)
    after = float(np.abs(np.asarray(policy.brain.learner.brain.efficacy)).max())
    assert np.isclose(after, 0.5 * before)
    assert policy.macs() > 0
