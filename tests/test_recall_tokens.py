"""Declared tokens (``recall_tokens``): which answers are tokens a slot can hold, so a lesson whose
answer is no token leaves the read as it was. The copy pathway (``recall_copy``) left in 0.96.0:
its name is refused at construction, under retune and in a saved memory."""
from __future__ import annotations

import json

import numpy as np
import pytest
from checkpoints import flat, metadata, write

import cadence as cd

TOKENS = 6
INPUTS = TOKENS + 3  # the tokens, a HEAR, an ASK and a NOISE flag
HEAR, ASK, NOISE = TOKENS, TOKENS + 1, TOKENS + 2


def composed(**options):
    options.setdefault("modules", (16,))
    return cd.Brain.compose(INPUTS, TOKENS, seed=3, episodic=False, recall=True,
                            recall_founder=True, **options)


def event(*on):
    x = np.zeros((1, INPUTS))
    x[0, list(on)] = 1.0
    return x


def test_the_copy_pathway_is_refused_by_name_with_what_replaces_it(tmp_path):
    for value in (0.0, 2.0):
        with pytest.raises(ValueError, match="recall_copy was retired in 0.96.0.*recall_gain"):
            composed(recall_copy=value, recall_tokens=range(TOKENS))
        with pytest.raises(ValueError, match="recall_copy was retired in 0.96.0"):
            composed().retune(recall_copy=value)
    with np.load(composed().save(tmp_path / "founder.npz"), allow_pickle=False) as data:
        arrays = flat({name: data[name].copy() for name in data.files})
    meta = metadata(arrays)
    meta["recall"]["copy"] = 2.0  # as a 0.95 memory with the pathway saved it
    arrays["generic"] = np.array(json.dumps(meta))
    write(tmp_path / "copy.npz", arrays)
    with pytest.raises(ValueError, match="recall_copy was retired in 0.96.0.*0.95.2"):
        cd.Brain.load(tmp_path / "copy.npz")
    meta["recall"]["copy"] = 0.0  # the value the memory keeps: a memory without the pathway
    arrays["generic"] = np.array(json.dumps(meta))
    write(tmp_path / "plain.npz", arrays)
    assert cd.Brain.load(tmp_path / "plain.npz").recall.tokens is None


def test_invalid_tokens_are_refused():
    with pytest.raises(ValueError, match="one observation coordinate per"):
        composed(recall_tokens=range(TOKENS - 1))
    with pytest.raises(ValueError, match="integer"):
        composed(recall_tokens=[0.5] * TOKENS)
    with pytest.raises(ValueError, match="sequence"):
        composed(recall_tokens="abc")


def test_tokens_declare_which_answers_are_tokens_and_add_no_synapse():
    brain = composed(recall_tokens=[0, 1, 2, 3, -1, -1], learning_surprise=True)
    assert brain.describe()["genes"]["recall_tokens"] == [0, 1, 2, 3, -1, -1]
    plain = composed(learning_surprise=True)
    np.testing.assert_array_equal(brain.brain.efficacy, plain.brain.efficacy)
    np.testing.assert_array_equal(brain.connectome.pre, plain.connectome.pre)
    brain.act(event(HEAR, 1), greedy=True)
    before = [np.array(m, copy=True) for m in brain.recall.head_maps[0]]
    assert brain.lesson(event(HEAR, ASK), [5])["address_rows"] == 0.0
    for a, b in zip(before, brain.recall.head_maps[0], strict=True):
        np.testing.assert_array_equal(a, b)
    assert brain.last_learning["demonstration_accepted_presentations"] == 1.0


def test_a_lesson_whose_answer_is_no_token_leaves_the_read_as_it_was():
    # actions 0 to 3 are tokens, 4 and 5 are not (a yes and a no): the read's lessons take
    # their target from the slots' contents, which hold no yes or no
    brain = composed(recall_tokens=[0, 1, 2, 3, -1, -1])
    brain.act(event(HEAR, 1), greedy=True)
    brain.act(event(NOISE, 3), greedy=True)
    before = [np.array(m, copy=True) for m in brain.recall.head_maps[0]]
    assert brain.lesson(event(HEAR, ASK), [5])["address_rows"] == 0.0
    for a, b in zip(before, brain.recall.head_maps[0], strict=True):
        np.testing.assert_array_equal(a, b)
    assert brain.lesson(event(HEAR, ASK), [3])["address_rows"] == 1.0  # a token: the read learns
    assert any(not np.array_equal(a, b)
               for a, b in zip(before, brain.recall.head_maps[0], strict=True))
    none = composed(recall_tokens=[-1] * TOKENS)  # no answer is a token: no lesson moves the read
    none.act(event(HEAR, 1), greedy=True)
    assert none.lesson(event(HEAR, ASK), [2])["address_rows"] == 0.0
    plain = composed()  # without tokens every taught row teaches the read
    plain.act(event(HEAR, 1), greedy=True)
    assert plain.lesson(event(HEAR, ASK), [5])["address_rows"] == 1.0


def test_under_surprise_a_lesson_whose_answer_is_no_token_teaches_every_row():
    x = np.concatenate([event(HEAR, 0), event(HEAR, 1)])
    taught = {}
    for name, tokens in (("tokens", list(range(TOKENS))), ("none", [-1] * TOKENS)):
        brain = composed(recall_tokens=tokens, learning_surprise=True)
        brain.act(x, greedy=True)
        answer = np.asarray(brain._greedy(  # noqa: SLF001  the act the lesson judges by
            brain._qualified(brain.stimulus(x), brain._activity(), operation="act"),  # noqa: SLF001
            brain._placed(x), brain._decision_key(brain._coded(x)),  # noqa: SLF001
        ))
        brain.lesson(x, answer)  # the label is the brain's own answer on both rows
        taught[name] = brain.last_learning["demonstration_accepted_presentations"]
    assert taught["tokens"] == 0.0  # answered as shown: no surprise, no lesson
    assert taught["none"] == 2.0  # no token answers: taught at every lesson, right or not
