"""Bounded recall storage preserves episode semantics and outstanding array snapshots."""

import os
import warnings
from contextlib import nullcontext
from copy import deepcopy

import numpy as np
import pytest

import cadence as cd
from cadence import stream


def two_slots():
    brain = cd.Brain.compose(
        2, 2, modules=(4,), episodic=False, recall=True, recall_founder=True,
        recall_temperature=0.25, recall_recency=0.0,
    )
    brain.act([[1.0, 0.0]], greedy=True)
    brain.act([[0.0, 1.0]], greedy=True)
    return brain


def attention_reference(recall, address):
    # Scalar dot products, independent of the memory's attention/cache implementation.
    logits = np.array([[sum(a * k for a, k in zip(query, key, strict=True))
                        / recall.temperature for key in slots]
                       for query, slots in zip(address, recall.keys, strict=True)])
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    return p / p.sum(axis=1, keepdims=True)


def test_retuned_temperature_and_saved_continuation_read_the_current_weights(tmp_path):
    brain = two_slots()
    address = brain.recall.keys[:, 0].copy()
    before = brain.recall.weights(address).copy()
    brain.retune(recall_temperature=1.0)
    expected = attention_reference(brain.recall, address)
    assert not np.array_equal(before, expected)
    np.testing.assert_array_equal(brain.recall.weights(address), expected)
    restored = cd.Brain.load(brain.save(tmp_path / "retuned.npz"))
    np.testing.assert_array_equal(restored.recall.weights(address), expected)


def test_current_key_contents_are_read_after_an_in_place_expert_edit():
    brain = two_slots()
    recall = brain.recall
    address = recall.keys[:, 0].copy()
    before = recall.weights(address).copy()
    recall.keys[:, 0] *= -1.0
    expected = attention_reference(recall, address)
    assert not np.array_equal(before, expected)
    np.testing.assert_array_equal(recall.weights(address), expected)


def test_content_reuse_is_bounded_and_checks_current_keys(monkeypatch):
    brain = two_slots()
    recall = brain.recall
    address = recall.keys[:, 0].copy()
    contractions = []
    original = stream._einsum

    def counted(*args):
        contractions.append(1)
        return original(*args)

    monkeypatch.setattr(stream, "_einsum", counted)
    recall.weights(address)
    recall.weights(address)
    brain.retune(recall_temperature=1.0)
    recall.weights(address)
    assert len(contractions) == 1
    recall.keys[:, 0] *= -1.0
    np.testing.assert_array_equal(recall.weights(address), attention_reference(recall, address))
    assert len(contractions) == 2
    for query in np.random.default_rng(8).normal(size=(40, 1, recall.address_width)):
        recall.weights(query)
        assert len(recall._contents) <= 4 * recall.heads
    recall.reset(0)
    assert not recall._contents and recall._content_keys is None


def test_an_operation_binds_the_bank_by_identity_and_releases_it(monkeypatch):
    brain = two_slots()
    recall = brain.recall
    address = recall.keys[:, 0].copy()
    contractions = []
    original = stream._einsum

    def counted(*args):
        contractions.append(1)
        return original(*args)

    monkeypatch.setattr(stream, "_einsum", counted)
    with recall._scope():
        first = recall.weights(address).copy()
        with recall._scope():  # a nested operation shares the outer one's binding
            np.testing.assert_array_equal(recall.weights(address), first)
        assert recall._content_keys is recall.keys
        np.testing.assert_array_equal(recall.weights(address), first)
        assert len(contractions) == 1
        # a write inside the operation replaces the bank; the next read scores the new one
        recall._observe(-recall.keys[:, -1], np.zeros((1, recall.value_width)))
        np.testing.assert_array_equal(
            recall.weights(address), attention_reference(recall, address)
        )
        assert len(contractions) == 2
    assert not recall._contents and recall._content_keys is None
    # between operations an in-place expert edit is read, as before
    recall.keys[:, 0] *= -1.0
    np.testing.assert_array_equal(recall.weights(address), attention_reference(recall, address))


def held_heads_brain(**options):
    return cd.Brain.compose(
        4, 3, modules=(8,), episodic=False, recall=True, recall_founder=True, recall_heads=2, recall_window=2,
        recall_temperature=0.25, recall_recency=0.0, seed=3, **options,
    )


def test_the_public_operations_read_on_the_bound_path(monkeypatch):
    brain = held_heads_brain(arousal=True)
    rng = np.random.default_rng(2)
    for x in rng.normal(size=(2, 1, 4)):  # slots to read
        brain.act(x, greedy=True)
    bound = []
    original = stream.Recall._content

    def spy(self, address, keys):
        bound.append(self._scope_token in stream._OPERATIONS)
        return original(self, address, keys)

    monkeypatch.setattr(stream.Recall, "_content", spy)
    for method in ("act", "lesson", "step", "live", "wait"):
        bound.clear()
        x = rng.normal(size=(1, 4))
        if method == "lesson":
            brain.lesson(x, [1])
        elif method == "step":
            brain.step(x, teacher=[1])
        elif method == "wait":
            brain.live(x)
            brain.wait(rng.normal(size=(1, 4)))
            bound[:] = bound[-1:]  # the wait's own read
        elif method == "act":
            brain.act(x, greedy=True)
        else:
            brain.live(x)
        assert bound and all(bound), method
    assert not stream._OPERATIONS


def test_operations_leave_every_public_operation_bit_identical(monkeypatch):
    def life(bound):
        if not bound:
            monkeypatch.setattr(stream.Recall, "_scope", lambda self: nullcontext())
        brain = held_heads_brain(
            arousal=True, recall_address_reward=0.05, recall_position=2, recall_capacity=4
        )
        rng = np.random.default_rng(5)
        acts = []
        for t in range(36):
            x = rng.normal(size=(1, 4))
            if t < 12:
                brain.lesson(x, rng.integers(0, 3, size=1))
                acts.append(brain.act(x, greedy=True))
            elif t == 12:
                acts.append(brain.live(x))
            elif t % 3 == 1:
                brain.wait(x)
            else:
                acts.append(brain.live(
                    x, reward=[float(rng.normal())], done=[t % 7 == 6],
                    decision_id=brain.decision_id,
                ))
        monkeypatch.undo()
        return np.array(acts), brain.recall.keys.copy(), brain.recall.head_map(0)[0].copy()

    for bound, unbound in zip(life(True), life(False), strict=True):
        np.testing.assert_array_equal(bound, unbound)


def test_an_in_place_edit_between_acts_reads_as_a_replaced_bank():
    rng = np.random.default_rng(9)
    xs = rng.normal(size=(6, 1, 4))
    answers = []
    for in_place in (True, False):
        brain = held_heads_brain()
        for x in xs[:3]:
            brain.act(x, greedy=True)
        if in_place:
            brain.recall.keys[:, 0] *= -1.0
        else:
            keys = brain.recall.keys.copy()
            keys[:, 0] *= -1.0
            brain.recall.keys = keys
        answers.append([brain.act(x, greedy=True) for x in xs[3:]])
    np.testing.assert_array_equal(*answers)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork is POSIX")
def test_a_forked_child_starts_without_its_parents_operations():
    import threading

    brain = two_slots()
    inside, release = threading.Event(), threading.Event()

    def hold():
        with brain.recall._scope():
            inside.set()
            release.wait(10)

    worker = threading.Thread(target=hold)
    worker.start()
    inside.wait(10)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # fork with a live thread
        pid = os.fork()
    if pid == 0:
        os._exit(0 if not stream._OPERATIONS else 1)
    release.set()
    worker.join()
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0


def test_the_lesson_warning_names_its_caller():
    brain = two_slots()
    x = np.array([[1.0, 0.0]])
    brain.act(x, greedy=True)
    with pytest.warns(RuntimeWarning, match="lesson on the observations") as caught:
        brain.lesson(x, [0])
    assert caught[0].filename == __file__


def test_a_partial_reset_inside_an_operation_is_read():
    brain = cd.Brain.compose(
        2, 2, modules=(4,), episodic=False, recall=True, recall_founder=True,
        recall_temperature=0.25, recall_recency=0.0,
    )
    brain.act([[1.0, 0.0], [0.0, 1.0]], greedy=True)
    brain.act([[0.0, 1.0], [1.0, 0.0]], greedy=True)
    recall = brain.recall
    address = recall.keys[:, -1].copy()
    with recall._scope():
        recall.weights(address)
        recall.reset(2, np.array([0]))  # new arrays: the stream holds no slot and reads silence
        inside = recall.weights(address).copy()
    np.testing.assert_array_equal(inside, recall.weights(address))
    np.testing.assert_array_equal(inside[0], 0.0)


def test_a_copy_taken_during_an_operation_reads_edits_after_it():
    brain = two_slots()
    address = brain.recall.keys[:, 0].copy()
    with brain.recall._scope():
        brain.recall.weights(address)
        twin = deepcopy(brain)
        branch = brain.recall.branch()
    for recall in (twin.recall, branch):
        recall.weights(address)
        recall.keys[:, 0] *= -1.0
        np.testing.assert_array_equal(
            recall.weights(address), attention_reference(recall, address)
        )
        recall.keys[:, 0] *= -1.0


@pytest.mark.parametrize("capacity", [1, 2, 7])
def test_successors_and_eviction_match_independent_stream_histories(capacity):
    recall = cd.Brain.compose(
        4, 2, modules=(8,), episodic=False, recall=True, recall_founder=True, recall_capacity=capacity,
    ).recall
    recall._empty(3)
    rng = np.random.default_rng(825)
    histories = [[] for _ in range(3)]
    for tick in range(25):
        # Not every stream holds every slot; episode endings can also differ per stream.
        closed = rng.random(3) < 0.3
        recall.ended = closed.copy()
        before = recall.snapshot()  # what a caller that holds the slots to restore them takes
        saved = {name: before[name].copy() for name in ("keys", "values", "nexts", "held")}
        keys, values = rng.normal(size=(2, 3, 4))
        if tick % 4 == 1:
            holds = rng.random(3) < 0.5
            recall._append(keys, holds, values)
        else:
            holds = np.ones(3, dtype=bool)
            recall._observe(keys, values)
            for row, history in enumerate(histories):
                if not closed[row]:
                    for slot in reversed(history):
                        if slot[3]:
                            slot[2] = values[row].copy()
                            break
        for row, history in enumerate(histories):
            history.append([keys[row], values[row], np.zeros(4), holds[row]])
            del history[:-capacity]
        for name, member in (("keys", 0), ("values", 1), ("nexts", 2), ("held", 3)):
            expected = np.array([[slot[member] for slot in history] for history in histories])
            np.testing.assert_array_equal(getattr(recall, name), expected, err_msg=name)
            np.testing.assert_array_equal(before[name], saved[name], err_msg="snapshot " + name)
