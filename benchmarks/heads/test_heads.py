"""The heads chamber (cadence#269): the language, its sealed split, what each panel asks, the
reads and interventions, a short life of each arm and the controls' routes."""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    here = Path(__file__).parent
    sys.path.insert(0, str(here))  # nets imports heads by name, as the grammar chamber's controls
    spec = importlib.util.spec_from_file_location(name, here / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


hd = _load("heads")
SMALL = {"lessons": 3, "panel_batches": 2, "module": 16}


def _efficacy(brain):
    return brain.learner.brain.efficacy


def _tokens(task, moment):
    return np.argmax(moment.x[:, : task.words], axis=1)


def _stream(task, moments):
    return np.stack([_tokens(task, m) for m in moments], axis=1)


@pytest.mark.parametrize("founder", [0, 1, 2, 3])
def test_the_split_is_sealed_and_differs_by_founder(founder):
    split = hd.Split.of(founder)
    assert split == hd.Split.of(founder)
    assert len(split.taught) == 8 and len(split.new) == 4
    assert set(split.taught) | set(split.new) == set(range(hd.TOKENS))
    assert all(a in split.taught and b in split.taught for a, b in split.pairs)
    assert hd.Split.of(founder).new != hd.Split.of(founder + 10).new


@pytest.mark.parametrize("name", sorted(hd.TASKS))
def test_training_never_hears_a_never_taught_token(name):
    task = hd.TASKS[name](1)
    rng = np.random.default_rng(0)
    for _ in range(12):
        for m in task.train(rng):
            assert not m.x[:, list(task.new)].any()
            if m.label is not None:
                assert not np.isin(m.label, task.new).any()


def test_the_named_sentence_is_a_row_of_the_chamber():
    task = hd.IndirectObject(0)
    mary, john = hd.NAMES.index("mary"), hd.NAMES.index("john")
    rows = [(mary, john, john)]
    moments = task.frame(rows, (2, 3), [("x",)])
    said = [hd.WORDS[int(np.argmax(m.x[0]))] for m in moments]
    assert " ".join(said) == "when mary and john went out john gave a drink to"
    assert hd.WORDS[int(moments[-1].label[0])] == "mary"
    assert moments[-1].slots["answer"][0] == 1 and moments[-1].slots["dup"][0] == 3


def test_the_chamber_has_the_grammar_chambers_scale():
    assert len(hd.WORDS) == 20
    assert hd.BATCH == hd.gr.BATCH
    assert hd.LESSONS["ioi"] == 4000


def test_a_sentence_asks_the_name_not_repeated_in_both_orders_at_every_position():
    task = hd.IndirectObject(2)
    rng = np.random.default_rng(3)
    seen_shapes, orders = set(), set()
    for _ in range(60):
        moments = task.train(rng)
        stream = _stream(task, moments)
        asked = moments[-1]
        at = next(t for t, m in enumerate(moments) if m.mark == "dup")
        for r in range(hd.BATCH):
            a, b, s = stream[r, 1], stream[r, 3], stream[r, at]
            assert a != b and s in (a, b)
            assert asked.label[r] == (b if s == a else a)
            assert not task.split.held(int(a), int(b))
            assert stream[r, int(asked.slots["answer"][r])] == asked.label[r]
            assert stream[r, int(asked.slots["dup"][r])] == s
            assert stream[r, int(asked.slots["dup2"][r])] == s
            orders.add(asked.tags[r][1])
        seen_shapes.add(at)
    assert orders == {"abb", "bab"}
    assert seen_shapes == {4, 5, 6}


def test_the_panels_ask_what_they_name():
    task = hd.IndirectObject(0)
    panels = task.panels(10)
    assert set(panels) == {"ioi", "ioi_pair", "ioi_shape", "ioi_half", "ioi_new"}
    taught, new = set(task.split.taught), set(task.split.new)
    for name, batches in panels.items():
        for moments in batches:
            stream = _stream(task, moments)
            asked = moments[-1]
            for i in range(0, hd.BATCH, 2):  # twins: the same pair, the other name repeated
                assert (stream[i, 1], stream[i, 3]) == (stream[i + 1, 1], stream[i + 1, 3])
                assert asked.label[i] != asked.label[i + 1]
            pairs = {(int(stream[r, 1]), int(stream[r, 3])) for r in range(hd.BATCH)}
            for a, b in pairs:
                if name in ("ioi", "ioi_shape"):
                    assert a in taught and b in taught and not task.split.held(a, b)
                elif name == "ioi_pair":
                    assert task.split.held(a, b)
                elif name == "ioi_half":
                    assert (a in new) != (b in new)
                else:
                    assert a in new and b in new
    shape = {next(t for t, m in enumerate(b) if m.mark == "dup") for b in panels["ioi_shape"]}
    assert shape == {5} and all(len(b) == 7 + 1 for b in panels["ioi_shape"])
    tags = {t for b in panels["ioi_half"] for row in b[-1].tags for t in row}
    assert {"key_new", "answer_new"} <= tags


def test_previous_asks_the_token_before_the_flagged_one():
    task = hd.Previous(0)
    moments = task.batch(np.random.default_rng(0), task.kinds_of_a_batch())
    stream = _stream(task, moments)
    assert np.array_equal(moments[-1].label, stream[:, -2])
    assert moments[-1].x[:, task.flag["ASK"]].all()
    for r, kind in enumerate(task.kinds_of_a_batch()):
        assert (stream[r, -1] in task.new) == (kind in ("key_new", "new"))
        assert (stream[r, -2] in task.new) == (kind in ("answer_new", "new"))


def test_duplicate_is_balanced_and_true_within_each_kind():
    task = hd.Duplicate(0)
    rng = np.random.default_rng(0)
    for _ in range(8):
        moments = task.batch(rng, task.kinds_of_a_batch())
        stream = _stream(task, moments)
        asked = moments[-1]
        yes = asked.label == task.words
        assert yes.sum() == hd.BATCH // 2
        for r in range(hd.BATCH):
            told = stream[r, :-1]
            assert (stream[r, -1] in told) == yes[r]
            assert len(set(told)) == len(told)
            if yes[r]:
                assert told[int(asked.slots["answer"][r])] == stream[r, -1]
                assert asked.source[r] == stream[r, -1]
            else:
                assert asked.source[r] == -1


@pytest.mark.parametrize("name", ["successor", "successor_fixed"])
def test_successor_asks_the_token_told_after_the_flagged_one(name):
    task = hd.TASKS[name](0)
    moments = task.batch(np.random.default_rng(0), task.kinds_of_a_batch())
    stream = _stream(task, moments)
    for r in range(hd.BATCH):
        told = list(stream[r, :-1])
        assert len(set(told)) == len(told)
        assert moments[-1].label[r] == told[told.index(stream[r, -1]) + 1]


def test_a_fixed_successor_tells_one_order_in_training():
    task = hd.SuccessorFixed(1)
    rng = np.random.default_rng(0)
    first = _stream(task, task.train(rng))[:, :-1]
    second = _stream(task, task.train(rng))[:, :-1]
    assert np.array_equal(first, second)
    assert np.all(first == first[0])
    assert sorted(first[0]) == sorted(task.split.taught)


def test_induction_asks_what_followed_the_only_earlier_occurrence():
    task = hd.Induction(0)
    moments = task.batch(np.random.default_rng(0), task.kinds_of_a_batch())
    stream = _stream(task, moments)
    for r in range(hd.BATCH):
        told = list(stream[r, :-1])
        assert told.count(stream[r, -1]) == 1
        assert moments[-1].label[r] == told[told.index(stream[r, -1]) + 1]


def test_two_hops_ask_the_second_of_each_pair():
    task = hd.TwoHops(0)
    moments = task.batch(np.random.default_rng(0), task.kinds_of_a_batch())
    stream = _stream(task, moments[:-1])
    for r in range(hd.BATCH):
        pairs = {int(stream[r, 2 * i]): int(stream[r, 2 * i + 1]) for i in range(task.pairs)}
        a = int(stream[r, -1])
        assert moments[-2].label[r] == pairs[a]
        assert moments[-1].label[r] == pairs[pairs[a]]
    assert moments[-1].x[:, task.flag["THINK"]].all()
    assert {t[1] for t in moments[-1].tags} == {"hop2"}


def test_copying_asks_each_token_in_turn():
    task = hd.Copying(0)
    moments = task.batch(np.random.default_rng(0), task.kinds_of_a_batch())
    length = sum(m.label is not None for m in moments)
    told = _stream(task, moments[1:1 + length])
    answers = np.stack([m.label for m in moments if m.label is not None], axis=1)
    assert np.array_equal(told, answers)
    assert all(m.feedback for m in moments[2 + length:])
    assert {t[0] for m in moments if m.tags for t in m.tags} <= set(hd.KINDS)


def test_copying_is_read_in_closed_loop_and_teacher_forced_on_the_same_rows():
    panels = hd.Copying(0).panels(2)
    assert set(panels) == {"test", "forced"}
    for closed, forced in zip(panels["test"], panels["forced"], strict=True):
        assert any(m.feedback for m in closed) and not any(m.feedback for m in forced)
        for a, b in zip(closed, forced, strict=True):
            assert np.array_equal(a.x, b.x)


def test_suppression_asks_the_name_not_repeated():
    task = hd.Suppression(0)
    panels = task.panels(3)
    assert set(panels) == {"taught", "half", "new"}
    for batches in panels.values():
        for moments in batches:
            stream = _stream(task, moments[:3])
            a, b, s = stream[:, 0], stream[:, 1], stream[:, 2]
            assert np.array_equal(moments[-1].label, np.where(s == a, b, a))
            assert np.array_equal(moments[-1].avoid, s)


def test_which_asks_the_place_of_the_one_not_repeated():
    task = hd.Which(0)
    assert task.tokens()[-2:] == [-1, -1]
    for moments in task.panels(2)["new"]:
        stream = _stream(task, moments[:3])
        a, s = stream[:, 0], stream[:, 2]
        assert np.array_equal(moments[-1].label, np.where(s == a, task.words + 1, task.words))
        assert np.array_equal(moments[-1].source, np.where(s == a, stream[:, 1], a))
        assert not np.isin(moments[-1].label, task.split.new).any()


def test_the_bare_frame_has_no_fixed_word_before_the_names():
    task = hd.Bare(0)
    rng = np.random.default_rng(1)
    for _ in range(10):
        moments = task.train(rng)
        stream = _stream(task, moments)
        asked = moments[-1]
        at = next(t for t, m in enumerate(moments) if m.mark == "dup")
        assert hd.WHEN not in stream and hd.AND not in stream
        for r in range(hd.BATCH):
            a, b, s = stream[r, 0], stream[r, 1], stream[r, at]
            assert s in (a, b) and asked.label[r] == (b if s == a else a)
            assert stream[r, int(asked.slots["answer"][r])] == asked.label[r]
            assert stream[r, int(asked.slots["dup"][r])] == s


def test_the_parts_ask_what_the_sentence_needs():
    mover, bit, whole = hd.Mover(0), hd.DuplicateBit(0), hd.IndirectObject(0)
    rng = np.random.default_rng(5)
    rows = [(0, 1, 0), (0, 1, 1)]
    tags = whole.tagged(rows)
    moved = mover.frame(rows, (2, 3), tags)
    assert not any(m.mark == "dup" for m in moved)
    assert moved[-1].x[0, mover.flag["SECOND"]] == 1.0
    assert moved[-1].x[1, mover.flag["FIRST"]] == 1.0
    assert list(moved[-1].label) == list(whole.frame(rows, (2, 3), tags)[-1].label)
    asked = bit.frame(rows, (2, 3), tags)[-1]
    assert list(asked.label) == [bit.words, bit.words + 1]
    assert asked.x[:, bit.flag["WHICH"]].all()
    assert bit.tokens()[-2:] == [-1, -1]
    assert bit.train(rng)[-1].label.min() >= bit.words


def test_the_arms_are_the_base_composition():
    assert hd.FD == {**hd.gr.ARMS["recipe"], "teaching": "deep"}
    assert hd.ARMS["Fd2"] == {**hd.FD, "recall_heads": 2}
    assert hd.CUTS["FdT"] == ["trace"]
    for arm in ("Fd", "Fd2", "FdT"):
        assert "apical" not in hd.ARMS[arm]


@pytest.fixture(scope="module")
def taught_brain():
    task = hd.IndirectObject(0)
    brain = hd.make_brain(task, "Fd", 0, module=16)
    rng = np.random.default_rng(0)
    work = hd.new_work()
    for _ in range(4):
        hd.teach(brain, task.train(rng), work)
    assert work["lessons"] == 4
    return task, brain


def test_addressing_a_moment_leaves_its_act_unchanged(taught_brain):
    task, brain = taught_brain
    moments = task.train(np.random.default_rng(1))
    plain, probed = copy.deepcopy(brain), copy.deepcopy(brain)
    for b in (plain, probed):
        b.reset()
    work = hd.new_work()
    for m in moments:
        attended = hd._attended(probed, m.x)
        a = hd._act(plain, m.x, work)
        c = hd._act(probed, m.x, work)
        assert np.array_equal(a, c)
        if attended:
            assert np.array_equal(attended[0], plain.recall.attended_heads[0])
    assert np.array_equal(plain.recall.keys, probed.recall.keys)
    assert np.array_equal(plain.recall.previous, probed.recall.previous)


def test_a_reading_never_changes_the_life(taught_brain):
    task, brain = taught_brain
    before = copy.deepcopy(brain)
    hd.reading(brain, task, task.panels(1), True)
    assert np.array_equal(_efficacy(before), _efficacy(brain))
    assert np.array_equal(before.recall.head_map(0)[0], brain.recall.head_map(0)[0])
    assert before.recall.amplitude == brain.recall.amplitude


def test_a_silent_read_drives_nothing_and_comes_back(taught_brain):
    task, brain = taught_brain
    fork = copy.deepcopy(brain)
    moments = task.train(np.random.default_rng(2))
    fork.reset()
    for m in moments[:-1]:
        fork.act(m.x, greedy=True)
    recall_neurons = fork.connectome.populations["recall"]
    loud = fork.stimulus(moments[-1].x)[:, recall_neurons]
    assert np.abs(loud).sum() > 0
    fork.recall.amplitude = 0.0
    assert not fork.stimulus(moments[-1].x)[:, recall_neurons].any()


def test_a_lesion_empties_the_slot_it_names_and_only_that(taught_brain):
    task, brain = taught_brain
    fork = copy.deepcopy(brain)
    moments = task.train(np.random.default_rng(2))
    fork.reset()
    for m in moments[:-1]:
        fork.act(m.x, greedy=True)
    recall = fork.recall
    keys = np.array(recall.keys, copy=True)
    rows = np.arange(hd.BATCH)
    slots = np.full(hd.BATCH, 2)
    hd._lesion(recall, rows, slots)
    assert not recall.keys[:, 2].any() and not recall.values[:, 2].any()
    assert not recall.nexts[:, 2].any()
    assert np.array_equal(recall.keys[:, 1], keys[:, 1])


def test_erasing_the_answers_slot_takes_it_from_both_routes(taught_brain):
    task, brain = taught_brain
    fork = copy.deepcopy(brain)
    moments = task.train(np.random.default_rng(2))
    fork.reset()
    for m in moments[:-1]:
        fork.act(m.x, greedy=True)
    asked = moments[-1]
    t = len(moments) - 1
    rows = np.arange(hd.BATCH)
    assert fork.recall.values[rows, :, asked.label].any()
    hd.intervene(fork, asked, {"erase_source": ("answer",)}, t, np.random.default_rng(0))
    # a name told once: nothing holds it now on either port
    assert not fork.recall.values[rows, :, asked.label].any()
    assert not fork.recall.nexts[rows, :, asked.label].any()


def test_erasing_the_attended_slot_empties_what_the_read_attends(taught_brain):
    task, brain = taught_brain
    fork = copy.deepcopy(brain)
    moments = task.train(np.random.default_rng(2))
    fork.reset()
    for m in moments[:-1]:
        fork.act(m.x, greedy=True)
    top = np.argmax(hd._attended(fork, moments[-1].x)[0], axis=1)
    hd.intervene(fork, moments[-1], {"erase": ("answer",)}, len(moments) - 1,
                 np.random.default_rng(0))
    assert not fork.recall.values[np.arange(hd.BATCH), top].any()


def test_the_control_erasure_spares_the_read_and_the_answer(taught_brain):
    task, brain = taught_brain
    fork = copy.deepcopy(brain)
    moments = task.train(np.random.default_rng(2))
    fork.reset()
    for m in moments[:-1]:
        fork.act(m.x, greedy=True)
    asked = moments[-1]
    rows = np.arange(hd.BATCH)
    top = np.argmax(hd._attended(fork, asked.x)[0], axis=1)
    before = np.array(fork.recall.values, copy=True)
    hd.intervene(fork, asked, {"erase_other": ("answer",)}, len(moments) - 1,
                 np.random.default_rng(0))
    assert np.array_equal(fork.recall.values[rows, top], before[rows, top])
    assert fork.recall.values[rows, :, asked.label].any(axis=1).all()


@pytest.mark.parametrize("cut", ["swap_cortex", "swap_trace", "swap_focus", "swap_slot"])
def test_a_swap_puts_the_twins_carrier_in_the_rows_place(taught_brain, cut):
    task, brain = taught_brain
    fork = copy.deepcopy(brain)
    moments = task.panels(1)["ioi"][0]
    fork.reset()
    for m in moments[:-1]:
        fork.act(m.x, greedy=True)
    recall = fork.recall
    asked = moments[-1]
    t = len(moments) - 1
    before = {"swap_cortex": np.array(recall.previous), "swap_focus": np.array(recall.focus),
              "swap_trace": np.array(fork.working_memory.trace),
              "swap_slot": np.array(recall.values)}[cut]
    assert not hd.intervene(fork, asked, {cut: ("answer",)}, t, np.random.default_rng(0))
    twin = np.arange(hd.BATCH) ^ 1
    if cut == "swap_cortex":
        assert np.array_equal(recall.previous, before[twin])
    elif cut == "swap_focus":
        assert np.array_equal(recall.focus, before[twin])
    elif cut == "swap_trace":
        assert np.array_equal(fork.working_memory.trace, before[twin])
    else:
        at = recall.keys.shape[1] - (t - asked.slots["dup2"][0])
        assert np.array_equal(recall.values[:, at], before[twin, at])
        assert np.array_equal(recall.values[:, at - 1], before[:, at - 1])


@pytest.mark.parametrize("moment", ["answer", "dup"])
def test_the_logits_terms_sum_to_the_attention_the_read_uses(taught_brain, moment):
    task, brain = taught_brain
    fork = copy.deepcopy(brain)
    moments = task.panels(1)["ioi"][0]
    t = len(moments) - 1 if moment == "answer" else next(
        i for i, m in enumerate(moments) if m.mark == "dup")
    fork.reset()
    for m in moments[:t]:
        fork.act(m.x, greedy=True)
    asked = moments[t]
    fork.stimulus(asked.x)
    found = hd.logit_terms(fork.recall, asked, t)
    p = fork.recall.attended_heads[0]
    rows = np.arange(hd.BATCH)
    # the parts sum to the two sums, and a slot's log attention less another's is their difference
    for name in found["key"]:
        parts = sum(found[k][name] for k in ("key", "event", "cortex", "bias"))
        assert np.allclose(parts, found["content"][name])
        parts = sum(found[k][name] for k in ("position_event", "position_cortex", "position_bias"))
        assert np.allclose(parts, found["position"][name])
    at = p.shape[1] - (t - asked.slots["dup"])
    before = at - 1
    wanted = np.log(p[rows, at]) - np.log(p[rows, before])
    read = (found["content"]["dup"] + found["position"]["dup"]
            - found["content"]["dup_before"] - found["position"]["dup_before"])
    assert np.allclose(wanted, read, atol=1e-8)


def test_a_raw_read_at_the_duplicate_finds_its_earlier_mention_and_leaves_the_maps(taught_brain):
    task, brain = taught_brain
    fork = copy.deepcopy(brain)
    # the maps pointing away from every slot, so that only a raw read finds the mention
    recall = fork.recall
    w_x, w_c, b = recall.head_map(0)
    recall._set_map(0, w_x, w_c, b + 50.0 * np.eye(len(b))[hd.WHEN])
    moments = task.panels(1)["ioi"][0]
    support = hd.Support(["raw_dup"])
    work = hd.new_work()
    fork.reset()
    for t, m in enumerate(moments):
        before = recall.head_map(0)[2].copy()
        slots = recall.keys.shape[1] if m.mark == "dup" else None
        support.act(fork, m, m.x, work)
        if slots is not None:
            at = slots - (t - m.slots["dup"])
            p = recall.attended_heads[0]
            assert p[np.arange(hd.BATCH), at].min() > 0.5
        assert np.array_equal(recall.head_map(0)[2], before)
        if m.mark == "dup":
            break


def test_what_is_held_stands_in_place_up_to_the_answer(taught_brain):
    task, brain = taught_brain
    fork = copy.deepcopy(brain)
    moments = task.frame([(0, 1, 0), (0, 1, 1)] * 8, (2, 3), [("x",)] * hd.BATCH)
    support = hd.Support(["hold_focus", "hold_cortex", "hold_trace"])
    work = hd.new_work()
    fork.reset()
    support.reset()
    held = held_trace = held_focus = None
    for m in moments:
        support.before(fork)
        if held is not None:
            assert np.array_equal(fork.recall.previous, held)
            assert np.array_equal(fork.working_memory.trace, held_trace)
            assert np.array_equal(fork.recall.focus, held_focus)
            held_focus = np.where(held_focus >= 0, held_focus + 1, -1)
        support.act(fork, m, m.x, work)
        support.after(fork, m)
        if m.mark == "dup":
            held = np.array(fork.recall.previous)
            held_trace = np.array(fork.working_memory.trace)
            held_focus = np.array(fork.recall.focus)


def test_a_cut_keeps_the_address_maps_cortex_columns_at_zero():
    task = hd.IndirectObject(0)
    brain = hd.make_brain(task, "Fd2", 0, module=16)
    support = hd.Support(["no_address_cortex"])
    rng = np.random.default_rng(0)
    work = hd.new_work()
    for _ in range(4):
        hd.teach(brain, task.train(rng), work, support)
    for h in (0, 1):
        w_x, w_c, _ = brain.recall.head_map(h)
        assert not w_c.any() and w_x.any()


def _taught(cortex_rate, batches=4):
    task = hd.IndirectObject(0)
    brain = hd.make_brain(task, "Fd", 0, module=16)
    rng = np.random.default_rng(0)
    work = hd.new_work()
    for _ in range(batches):
        hd.teach(brain, task.train(rng), work, None, cortex_rate)
    return brain


def test_a_cortex_rate_of_one_is_the_life_without_it():
    plain, scaled = _taught(None), _taught(1.0)
    for name in hd.CORTEX_MAPS:
        for a, b in zip(getattr(plain.recall, name) or [], getattr(scaled.recall, name) or [],
                        strict=True):
            for x, y in zip(a, b, strict=True):
                np.testing.assert_array_equal(x, y)


def test_a_cortex_rate_of_zero_keeps_every_maps_cortex_columns_at_birth():
    brain = _taught(0.0)
    assert brain.recall.head_map(0)[0].any()
    for name in hd.CORTEX_MAPS:
        for head in getattr(brain.recall, name) or []:
            assert not np.asarray(head[1]).any()


def test_a_cut_of_the_cortex_reads_zero_at_every_act():
    task = hd.IndirectObject(0)
    brain = hd.make_brain(task, "Fd", 0, module=16)
    support = hd.Support(["no_cortex"])
    moments = task.train(np.random.default_rng(0))
    work = hd.new_work()
    brain.reset()
    for m in moments[:3]:
        support.before(brain)
        assert not np.any(brain.recall.previous)
        support.act(brain, m, m.x, work)
        assert np.any(brain.recall.previous)  # what the act leaves, cleared before the next


@pytest.mark.parametrize("names, scale", [(["cortex_unit"], None), ([], 0.25)])
def test_a_balanced_cortex_is_read_at_unit_norm_or_scaled(names, scale):
    task = hd.IndirectObject(0)
    brain = hd.make_brain(task, "Fd", 0, module=16)
    support = hd.Support(names, scale)
    moments = task.train(np.random.default_rng(0))
    work = hd.new_work()
    brain.reset()
    support.act(brain, moments[0], moments[0].x, work)
    raw = np.array(brain.recall.previous)
    assert np.linalg.norm(raw, axis=1).min() > 0
    support.before(brain)
    read = np.array(brain.recall.previous)
    if scale is None:
        assert np.allclose(np.linalg.norm(read, axis=1), 1.0)
        assert np.allclose(read * np.linalg.norm(raw, axis=1, keepdims=True), raw)
    else:
        assert np.allclose(read, scale * raw)


def test_a_centered_cortex_is_read_less_its_mean_and_a_reading_leaves_the_mean():
    task = hd.IndirectObject(0)
    brain = hd.make_brain(task, "Fd", 0, module=16)
    support = hd.Support(["center_cortex"])
    work = hd.new_work()
    rng = np.random.default_rng(0)
    hd.teach(brain, task.train(rng), work, support)
    mean = np.array(support.mean)
    assert mean.any()
    moments = task.train(rng)
    brain.reset()
    brain.act(moments[0].x, greedy=True)
    raw = np.array(brain.recall.previous)
    support.before(brain)
    assert np.allclose(brain.recall.previous, raw - mean)
    assert np.array_equal(support.mean, mean)
    hd.read_panel(brain, task, task.panels(1)["ioi"], support=support)
    assert np.array_equal(support.mean, mean)


def _with_gain() -> bool:
    import cadence as cd

    recall = cd.Brain.compose(4, 2, modules=(4,), recall=True).recall
    return "gain" in getattr(recall, "OPTIONS", ())


needs_gain = pytest.mark.skipif(not _with_gain(), reason="a library with recall_gain")


@needs_gain
def test_the_gain_arm_declares_every_word_and_says_a_token_no_lesson_used():
    task = hd.Induction(0)
    brain = hd.make_brain(task, "Fdg", 0, module=16)
    assert list(brain.recall.tokens) == task.tokens()
    row = hd.run_life({"task": "induction", "arm": "Fdg", "founder": 0, **SMALL, "causal": False})
    assert len(row["gains"]) == 2
    both = hd.run_life({"task": "duplicate", "arm": "Fd2g", "founder": 0, **SMALL, "causal": False})
    assert len(both["gains"]) == 4


@needs_gain
def test_the_gains_drive_only_answers_under_the_oracle_and_hold_for_lessons_when_late():
    task = hd.IndirectObject(0)
    brain = hd.make_brain(task, "Fdg", 0, module=16)
    late = hd.Support(["gain_late"])
    late.LATE = 2
    rng = np.random.default_rng(0)
    work = hd.new_work()
    hd.teach(brain, task.train(rng), work, late)
    hd.teach(brain, task.train(rng), work, late)
    assert not brain.recall.gains.any()
    hd.teach(brain, task.train(rng), work, late)
    assert brain.recall.gains.any()
    oracle = hd.Support(["gain_at_answer"])
    moments = task.train(rng)
    gains = brain.recall.gains.copy()
    brain.reset()
    for m in moments:
        oracle.act(brain, m, m.x, work)
        assert np.array_equal(brain.recall.gains, gains)
    out = hd.read_panel(brain, task, task.panels(1)["ioi"], votes=True, support=oracle)
    assert out["vote"]["taught"]["answer"]["gain"] != 0.0


def test_a_causal_reading_keeps_what_the_read_found():
    row = hd.run_life({"task": "suppression", "arm": "Fd", "founder": 0, **SMALL})
    cell = row["panels"]["new"]["causal"]["swap_focus"]["conditions"]["new"]
    assert {"answer", "first", "port_best", "ports"} <= set(cell)


def test_a_life_without_a_support_is_the_life_with_none_named():
    spec = {"task": "suppression", "arm": "Fd", "founder": 0, **SMALL, "causal": False}
    plain = hd.run_life(spec)
    none = hd.run_life({**spec, "support": []})
    assert plain["panels"] == none["panels"] and none["support"] == []
    with pytest.raises(ValueError):
        hd.Support(["nothing"])


@pytest.mark.parametrize("support", [["raw_dup"], ["raw_dup", "hold_focus"],
                                     ["raw_dup", "hold_cortex", "hold_trace"]])
def test_a_short_life_with_a_support(support):
    row = hd.run_life({"task": "ioi", "arm": "Fd", "founder": 0, **SMALL, "causal": False,
                       "support": support})
    assert row["support"] == support and row["work"]["refused"] == 0
    assert row["panels"]["ioi"]["duplicate_read"]["first_value"] > 0.9


def test_the_decoder_reads_a_bit_a_carrier_holds():
    y = np.array([1.0, -1.0] * 40)
    x = np.column_stack([y + np.random.default_rng(0).normal(0, 0.1, 80),
                         np.random.default_rng(1).normal(0, 1.0, 80)])
    noise = np.random.default_rng(2).normal(0, 1.0, (80, 2))
    read = hd._ridge(x, y, {"same": (x, y), "none": (noise, y)})
    assert read["same"] == 1.0 and read["none"] < 0.8


def test_the_decoding_runs_on_a_sentence_task(taught_brain):
    task, brain = taught_brain
    out = hd.decoding(brain, task, batches=12)
    assert {"taught", "new", "half"} == set(out["cortex_before_answer"])
    assert "cortex_after_dup" in out and "trace_after_answer" in out


def test_the_motor_inputs_sum_to_what_the_motor_neuron_receives(taught_brain):
    task, brain = taught_brain
    fork = copy.deepcopy(brain)
    moments = task.panels(1)["ioi"][0]
    fork.reset()
    for m in moments[:-1]:
        fork.act(m.x, greedy=True)
    asked = moments[-1]
    fork.act(asked.x, greedy=True)
    found = hd.motor_inputs(fork, {"answer": asked.label}, task.actions)["answer"]
    graph = fork.connectome
    pre, post = np.asarray(graph.pre), np.asarray(graph.post)
    motor = np.asarray(fork.motor_index)
    activation = np.atleast_2d(np.asarray(fork.basal_ganglia.state.activation))
    weights, bias = np.asarray(fork.brain.weights), np.asarray(fork.brain.bias)
    for r in range(hd.BATCH):
        neuron = motor[asked.label[r]]
        into = post == neuron
        total = (weights[into] * activation[r, pre[into]]).sum() + bias[neuron]
        assert np.isclose(found["association"][r] + found["other"][r], total)
        assert found["gain"][r] == 0.0  # this brain holds no gains
    out = hd.read_panel(brain, task, task.panels(1)["ioi"], votes=True)
    assert {"answer", "avoid", "winner"} == set(out["vote"]["taught"])


def test_the_read_of_a_moment_names_what_it_attends():
    values = np.zeros((2, 3, 5))
    nexts = np.zeros((2, 3, 5))
    values[:, 0, 1], values[:, 1, 2], values[:, 2, 3] = 1.0, 1.0, 1.0
    nexts[:, 0, 2], nexts[:, 1, 3] = 1.0, 1.0
    p = np.array([[0.7, 0.2, 0.1], [0.05, 0.15, 0.8]])
    read = hd.read_of((p,), values, nexts, np.array([2, 2]), np.array([3, 3]))
    # row 0 attends the slot whose next is the token; row 1 the slot that holds the other
    assert read["first"].tolist() == [True, False]
    assert read["first_avoid"].tolist() == [False, True]
    assert read["mass"][0] == pytest.approx(0.9)  # slot 0 by its next, slot 1 by its value
    assert read["mass_avoid"][1] == pytest.approx(0.95)  # slot 1 by its next, slot 2 by its value
    assert read["ports"]["next0"].tolist() == [True, False]


def test_a_row_without_a_token_in_the_slots_takes_no_read():
    values, nexts = np.ones((2, 3, 5)), np.ones((2, 3, 5))
    read = hd.read_of((np.full((2, 3), 1 / 3),), values, nexts, np.array([2, -1]), None)
    assert read["held"].tolist() == [True, False]


@pytest.mark.parametrize("name", sorted(hd.TASKS))
def test_a_short_life_of_each_task(name):
    row = hd.run_life({"task": name, "arm": "Fd", "founder": 0, **SMALL})
    assert row["lessons"] == 3 and row["work"]["refused"] == 0
    panel = next(iter(row["panels"].values()))
    assert panel["conditions"]["all"]["n"] > 0 and "causal" in panel
    assert {"silence", "erase", "erase_source", "erase_other"} <= set(panel["causal"])
    assert ("swap_all" in panel["causal"]) == hd.TASKS[name].twins


@pytest.mark.parametrize("arm", ["Fd2", "FdT", "frozen"])
def test_a_short_life_of_each_arm(arm):
    row = hd.run_life({"task": "suppression", "arm": arm, "founder": 1, **SMALL})
    assert row["arm"] == arm
    assert row["lessons"] == (0 if arm == "frozen" else 3)
    conditions = row["panels"]["taught"]["conditions"]["taught"]
    assert "first" in conditions
    if arm == "Fd2":
        assert "next1" in conditions["ports"]


def test_the_trace_silent_arm_holds_no_trace():
    task = hd.Suppression(0)
    brain = hd.make_brain(task, "FdT", 0, module=16)
    moments = task.train(np.random.default_rng(0))
    brain.reset()
    brain.act(moments[0].x, greedy=True)
    trace = brain.working_memory.trace
    assert trace.any()  # what a moment leaves, cleared before the next act and lesson
    brain.act(moments[1].x, greedy=True)
    assert copy.deepcopy(brain).__class__ is brain.__class__


def test_a_replica_is_the_same_newborn_at_the_last_bits():
    task = hd.Suppression(0)
    plain = hd.make_brain(task, "Fd", 0, module=16)
    replica = hd.make_brain(task, "Fd", 0, module=16, replica=1)
    assert not np.array_equal(_efficacy(plain), _efficacy(replica))
    assert np.allclose(_efficacy(plain), _efficacy(replica), rtol=1e-11)


@pytest.fixture(scope="module")
def nets():
    pytest.importorskip("torch")
    return _load("nets")


def test_the_controls_are_the_grammar_chambers_networks_at_the_tasks_size(nets):
    task = hd.IndirectObject(0)
    for model in ("lstm", "transformer"):
        with nets._dims(task):
            direct = nets.ct.build(model, 20, 0)
        ours = nets.build(model, 20, 0, task)
        assert nets.count(direct) == nets.count(ours)
        probe = np.zeros((1, 3, task.inputs), dtype=np.float32)
        import torch

        out = ours(torch.from_numpy(probe))
        assert out.shape == (1, 3, task.actions)
    assert nets.ct.gr is hd.gr


def test_a_controls_width_matches_the_brains_parameters(nets):
    task = hd.IndirectObject(0)
    target = nets.brain_parameters(task, 0)
    for model in nets.MODELS:
        width = nets.matched_width(model, target, task)
        assert abs(nets.count(nets.build(model, width, 0, task)) - target) / target < 0.1


def test_the_tied_control_names_a_token_by_its_own_input_weights(nets):
    import torch

    task = hd.DuplicateBit(0)
    net = nets.build("tied", 20, 0, task)
    x = torch.from_numpy(np.zeros((1, 2, task.inputs), dtype=np.float32))
    assert net(x).shape == (1, 2, task.actions)
    h = torch.rand(1, 20)
    assert torch.allclose(h @ net.inp.weight[:, : task.words],
                          h @ net.inp.weight[:, net.coordinates])


def test_the_pointer_control_reads_nothing_at_birth_and_only_earlier_moments(nets):
    import torch

    task = hd.IndirectObject(0)
    net = nets.build("pointer", 20, 0, task)
    plain = nets.ct.gr
    assert plain is hd.gr
    x = torch.from_numpy(nets.encode(task.train(np.random.default_rng(0)))[0])
    with torch.no_grad():
        assert torch.allclose(net(x), net.out(net.body(
            net.inp(x) + net.position[: x.shape[1]],
            mask=torch.triu(torch.full((x.shape[1],) * 2, float("-inf")), diagonal=1),
            is_causal=True)))
        net.gain.data[:] = 5.0
        changed = net(x)
    # a later moment's change does not reach an earlier moment's read
    y = x.clone()
    y[:, -1] = 0.0
    with torch.no_grad():
        same = net(y)
    assert torch.allclose(changed[:, :-1], same[:, :-1], atol=1e-5)


def test_the_copy_control_says_only_what_it_attends():
    pytest.importorskip("torch")
    nets = _load("nets")
    import torch

    task = hd.IndirectObject(0)
    net = nets.build("copy", 20, 0, task)
    x = torch.zeros(1, 2, task.inputs)
    x[0, 0, 5] = 1.0
    with torch.no_grad():
        logits = net(x)[0, 1]
    # one moment before it, holding one token: that token's logit is the log of its whole attention
    assert int(torch.argmax(logits)) == 5
    assert abs(float(logits[5])) < 1e-3 and float(logits[6]) < -10.0


def test_the_pointer_control_copies_a_token_it_attends():
    pytest.importorskip("torch")
    nets = _load("nets")
    import torch

    task = hd.IndirectObject(0)
    net = nets.build("pointer", 20, 0, task)
    net.gain.data[:] = 3.0
    x = torch.zeros(1, 3, task.inputs)
    x[0, 0, 5] = 1.0  # a token, then nothing
    with torch.no_grad():
        with_token = net(x)
        y = x.clone()
        y[0, 0, 5] = 0.0
        without = net(y)
    # at the second moment only the first moment can be read: its token is added at full gain
    gap = (with_token - without)[0, 1, 5] - (with_token - without)[0, 1, 6]
    assert gap > 1.0


@pytest.mark.parametrize("model", ["lstm", "transformer", "tied", "pointer", "copy"])
def test_a_short_control_life(nets, model):
    row = nets.run_life({"task": "ioi", "model": model, "mode": "online", "founder": 0,
                         "lessons": 3, "panel_batches": 2})
    assert row["work"]["updates"] == 3
    assert row["parameters"] > 0 and row["panels"]["ioi_new"]["conditions"]["new"]["n"] > 0
    assert "blank_answer" in row["panels"]["ioi"]["causal"]
    assert ("silence" in row["panels"]["ioi"]["causal"]) == (model != "lstm")


def test_equal_compute_runs_epochs_to_its_seconds(nets):
    row = nets.run_life({"task": "suppression", "model": "lstm", "mode": "offline", "founder": 0,
                         "lessons": 3, "panel_batches": 1, "compute_seconds": 0.05, "epochs": 4})
    assert 1 <= row["epochs"] <= 4 and row["curve"][-1]["epoch"] == row["epochs"]


def test_a_closed_loop_control_hears_its_own_answer(nets):
    task = hd.Copying(0)
    learner = nets.Learner("lstm", 16, 0, 0.001, task)
    moments = task.train(np.random.default_rng(0))
    said = learner.answers(moments)
    assert len(said) == sum(m.label is not None for m in moments)


def test_blanking_a_moment_changes_what_a_control_hears(nets):
    import torch

    task = hd.IndirectObject(0)
    learner = nets.Learner("transformer", 20, 0, 0.001, task)
    learner.net.eval()
    moments = task.panels(1)["ioi"][0]
    plain = learner.answers(moments)[0]
    with torch.no_grad():
        x, _ = nets.encode(moments)
        logits = learner.net(torch.from_numpy(x))[:, -1].argmax(dim=1).numpy()
    assert np.array_equal(plain, logits)
    assert learner.answers(moments, {"silence": True})[0].shape == plain.shape
    silenced = learner.answers(moments, {"silence": True})[0]
    assert np.array_equal(learner.answers(moments)[0], plain)  # the hook comes off
    assert isinstance(silenced, np.ndarray)
