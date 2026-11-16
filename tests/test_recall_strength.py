"""The strength law on the recall memory (cadence#336): every slot keeps a depth from
repetition, surprise and use, and a full bank gives up its weakest slot outside the protected
newest share in place of its oldest.

``Brain.compose(recall_strength=True)``. Unset, a brain is the plain brain to the bit; set, a
refused attempt and an imagined branch leave the bank as it was, and a saved brain continues with
the same depths."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from checkpoints import flat, metadata, write

import cadence as cd

WIDTH = 6
E = np.eye(WIDTH)
A, B, C, D = (E[[i]] for i in range(4))
AB = (E[[0]] + E[[1]])  # holds A and more: at cosine .71 to it, which tells nothing again


def compose(**options):
    return cd.Brain.compose(WIDTH, 3, modules=(8,), seed=3, backend="cpu", episodic=False,
                            **options)


def strong(capacity: int = 8, **options):
    return compose(recall_strength=True, recall_capacity=capacity, **options)


def told(brain, *moments) -> None:
    for x in moments:
        brain.act(x, greedy=True)


def others(count: int, seed: int = 0, width: int = WIDTH) -> list[np.ndarray]:
    """Moments no two of which are one moment told again: random signs on every input."""
    rng = np.random.default_rng(seed)
    out: list[np.ndarray] = []
    while len(out) < count:
        x = rng.choice([-1.0, 1.0], size=(1, width)) * rng.uniform(0.2, 1.0, size=(1, width))
        unit = x / np.linalg.norm(x)
        if all(float((unit @ (y / np.linalg.norm(y)).T)[0, 0]) < 0.9 for y in out) and (
            width != WIDTH or all(float(np.abs(unit @ z.T).max()) < 0.9 for z in (A, B, C, D))
        ):
            out.append(x)
    return out


def held(recall, x, row: int = 0, width: int = WIDTH) -> list[int]:
    """The slots of stream ``row`` that hold the moment ``x``, oldest first."""
    assert x.shape == (1, width)
    same = (recall.keys[row] == x[0]).all(axis=1) & recall.held[row]
    return [int(i) for i in np.flatnonzero(same)]


def law(events, capacity, protected, *, boundary=False, recency=0.0, surprise=None,
        again=0.95, kept=8.0, temperature=0.25):
    """The law written out for one stream of a newborn memory, whose read is the content's and
    the recency's alone: the keys held and their depths after ``events``, oldest first."""
    keys = [np.zeros(WIDTH)] if boundary else []
    depth = [1.0] if boundary else []
    stay = 1.0 - 1.0 / (cd.Recall.HOLD * capacity)
    for t, x in enumerate(events):
        x = np.asarray(x, dtype=float).reshape(-1)
        if keys:
            bank = np.array(keys)
            ages = np.arange(len(keys) - 1, -1, -1)
            logits = bank @ x / temperature - recency * ages
            p = np.exp(logits - logits.max())
            p /= p.sum()
            if t == 0:  # a stream begins at its boundary: the moment's read found no slot
                p = np.zeros(len(keys))
            depth = [d if d >= kept else d * stay for d in depth]  # the moment passes
            depth = [d + w * max(1.0 - d, 0.0) for d, w in zip(depth, p, strict=True)]  # use
            lengths = np.linalg.norm(bank, axis=1) * np.linalg.norm(x)
            again_at = [i for i in range(len(keys))
                        if lengths[i] > 0 and bank[i] @ x >= again * lengths[i]]
            if again_at:  # told again: the deepest of them, the oldest among equals
                depth[max(again_at, key=lambda i: (depth[i], -i))] += 1.0
        if len(keys) == capacity:
            pool = range(max(capacity - protected, 1))
            gone = min(pool, key=lambda i: (depth[i], i))
            del keys[gone], depth[gone]
        keys.append(x)
        depth.append(1.0 + (0.0 if surprise is None else float(surprise[t])))
    return np.array(keys), np.array(depth)


def life(brain, moments: int = 40, seed: int = 1) -> list[int]:
    """A continuing life under ``live``: the action paid when it is the count of active inputs
    modulo three. Gives the actions."""
    rng = np.random.default_rng(seed)
    reward, actions = None, []
    for _ in range(moments):
        x = (rng.random((1, WIDTH)) < 0.4).astype(float)
        action = brain.live(x, reward=None if reward is None else np.array([reward]))
        reward = float(action[0] == int(x[0].sum()) % 3)
        actions.append(int(action[0]))
    return actions


def stepped(brain, moments: int = 24, seed: int = 3) -> list[int]:
    """The same world under ``step`` with a teacher at every third moment, on two fresh
    streams."""
    brain.reset()
    rng = np.random.default_rng(seed)
    reward, actions = None, []
    for moment in range(moments):
        x = (rng.random((2, WIDTH)) < 0.4).astype(float)
        teacher = x.sum(axis=1).astype(int) % 3 if moment % 3 == 2 else None
        action = brain.step(x, reward=reward, teacher=teacher)
        reward = (action == x.sum(axis=1).astype(int) % 3).astype(float)
        actions.extend(int(a) for a in action)
    return actions


def saved(brain, folder: Path, name: str) -> dict[str, np.ndarray]:
    path = brain.save(folder / name)
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def record(arrays) -> dict:
    return json.loads(str(arrays["brain"]))


# -- unset: the plain brain


@pytest.mark.parametrize("compiled", [False, True], ids=["moments in turn", "compiled moment"])
def test_unset_a_brain_is_the_plain_brain_to_the_bit(tmp_path, compiled):
    """Weights, a life's actions and the saved file's arrays, with a bank that fills and drops."""
    options = {"learning_compiled_moment": True} if compiled else {}
    plain = cd.Brain.compose(WIDTH, 3, seed=2, backend="cpu", arousal=True, recall_capacity=8,
                             **options)
    off = cd.Brain.compose(WIDTH, 3, seed=2, backend="cpu", arousal=True, recall_capacity=8,
                           recall_strength=False, recall_again=0.95, recall_kept=8.0, **options)
    assert not off.recall.strength and plain.recall.depth is None and off.recall.depth is None
    assert life(plain) == life(off)
    assert stepped(plain) == stepped(off)
    assert plain.recall.slots == 8 and plain.recall.writes > 16
    assert np.array_equal(plain.brain.efficacy, off.brain.efficacy)
    assert np.array_equal(plain.brain.bias, off.brain.bias)
    left, right = saved(plain, tmp_path, "plain.npz"), saved(off, tmp_path, "off.npz")
    assert sorted(left) == sorted(right)
    for name in left:
        assert np.array_equal(left[name], right[name]), name
    assert "memories/recall/depth" not in left and "features" not in record(left)
    assert record(left)["format"] == "cadence-brain/1"
    assert "strength" not in record(left)["memories"]["recall"]
    genes = plain.describe()["genes"]
    assert not {"recall_strength", "recall_again", "recall_kept"} & set(genes)
    assert plain.describe() == off.describe()


def test_set_a_life_that_fills_no_bank_is_the_plain_life_to_the_bit(tmp_path):
    """Until a bank gives up a slot, the law only keeps the depths: the same actions, weights and
    slots, and a file that holds the depths beside them."""
    plain = cd.Brain.compose(WIDTH, 3, seed=2, backend="cpu", arousal=True)
    deep = cd.Brain.compose(WIDTH, 3, seed=2, backend="cpu", arousal=True, recall_strength=True)
    assert deep.recall.strength and deep.recall.capacity == 128 and deep.recall.protected == 64
    assert life(plain, 50) == life(deep, 50)
    assert stepped(plain) == stepped(deep)
    assert np.array_equal(plain.brain.efficacy, deep.brain.efficacy)
    left, right = saved(plain, tmp_path, "plain.npz"), saved(deep, tmp_path, "deep.npz")
    assert sorted({*left, "memories/recall/depth"}) == sorted(right)
    for name in left:
        if name != "brain":
            assert np.array_equal(left[name], right[name]), name
    depth = right["memories/recall/depth"]
    assert depth.shape == deep.recall.held.shape and (depth[deep.recall.held] > 0).all()
    there, here = record(right), record(left)
    assert there.pop("format") == "cadence-brain/5" and here.pop("format") == "cadence-brain/1"
    assert there.pop("features") == ["recall_strength"]
    assert there["memories"]["recall"].pop("strength") is True
    for told_by in (there, here):  # the composition's record of its own genes names the option
        told_by.pop("composition", None)
    assert there == here


def wide(capacity: int = 16, **options):
    return cd.Brain.compose(12, 3, modules=(8,), seed=3, backend="cpu", episodic=False,
                            recall_strength=True, recall_capacity=capacity, recall_founder=True,
                            recall_recency=0.0, **options)


def apart(count: int, seed: int) -> list[np.ndarray]:
    """Moments of twelve inputs that share nothing with the first three, no two alike."""
    return [np.concatenate([np.zeros((1, 3)), x], axis=1) for x in others(count, seed, width=9)]


# -- the law


@pytest.mark.parametrize("founder", [True, False], ids=["founder memory", "a new brain's memory"])
def test_the_bank_is_the_law_written_out_moment_for_moment(founder):
    """A depth of one a telling, kept at ``1 - 1 / (8 capacities)`` a moment below the kept
    depth; a read renews what it reads toward one telling; a moment told again deepens the
    deepest slot that holds it; a full bank gives up its weakest slot outside the protected
    newest share, the oldest among equals."""
    options = {"recall_founder": True} if founder else {}
    brain = strong(capacity=8, **options)
    recall = brain.recall
    assert recall.protected == 4 and cd.Recall.HOLD == 8.0 and cd.Recall.PROTECT == 0.5
    assert (recall.again, recall.kept) == (0.95, 8.0)
    rng = np.random.default_rng(5)
    alphabet = [A, B, C, D, AB, *others(6, seed=1)]
    events: list[np.ndarray] = []
    middle = False
    for _ in range(90):
        events.append(alphabet[int(rng.integers(len(alphabet)))])
        before = None if recall.depth is None or not recall.slots else recall.keys[0].copy()
        told(brain, events[-1])
        keys, depth = law(events, 8, 4, boundary=not founder,
                          recency=recall.recency)
        np.testing.assert_array_equal(recall.keys[0], keys)
        np.testing.assert_allclose(recall.depth[0], depth, rtol=1e-9, atol=0)
        if before is not None and len(before) == 8:
            middle |= not np.array_equal(before[1:], recall.keys[0, :-1])
        # the protected share: the newest moments in their order, each at its age
        newest = min(len(events), 4)
        np.testing.assert_array_equal(
            recall.keys[0, -newest:], np.concatenate(events[-newest:])
        )
    assert middle  # a slot other than the oldest was given up on the way
    assert recall.depth.shape == (1, 8) and recall.slots == 8 and recall.held.all()
    np.testing.assert_array_equal(recall.ages(), np.arange(7, -1, -1))


def test_a_moment_told_again_outlives_the_moments_told_once_and_then_fades():
    brain = strong(capacity=8, recall_founder=True)
    recall = brain.recall
    told(brain, A, A, *others(5, seed=2))
    assert held(recall, A) == [0, 1] and recall.depth[0, 0] > 1.8 > 1.0 > recall.depth[0, 1]
    spans = []
    for x in others(120, seed=3):
        told(brain, x)
        spans.append(bool(held(recall, A)))
        if spans[-1]:
            assert held(recall, A)[0] == 0  # the first telling, at the oldest place
    gone = spans.index(False)
    # told once it leaves with the eighth moment after it; told twice it stays for several
    # capacities, and it fades: a depth of two is kept over eight capacities
    assert 4 * 8 < gone < 8 * 8 and not any(spans[gone:])
    plain = compose(recall_capacity=8, recall_founder=True)
    told(plain, A, A, *others(8, seed=2))
    assert not held(plain.recall, A)


def test_a_moment_told_past_the_kept_depth_stays_and_a_bank_of_kept_slots_gives_up_its_weakest():
    brain = strong(capacity=8, recall_founder=True)
    recall = brain.recall
    fillers = iter(others(260, seed=4))
    told(brain, *[A] * 9)
    assert recall.depth[0, held(recall, A)[0]] >= recall.kept
    whole = float(recall.depth[0, held(recall, A)[0]])
    told(brain, *[next(fillers) for _ in range(200)])
    assert held(recall, A) == [0] and recall.depth[0, 0] == whole  # a kept depth stays whole
    for x in (B, C):  # two more moments told past the kept depth
        told(brain, *[x] * 9, *[next(fillers) for _ in range(8)])
    assert [held(recall, x) for x in (A, B, C)] == [[0], [1], [2]]
    assert (recall.depth[0, :3] >= recall.kept).all() and (recall.depth[0, 3:] < 1.1).all()
    # the kept slots hold all but one place outside the protected share: a moment that leaves
    # the share below the kept depth is the weakest there, however often it is told after
    told(brain, *[D] * 9, *[next(fillers) for _ in range(8)])
    assert not held(recall, D) and [held(recall, x) for x in (A, B, C)] == [[0], [1], [2]]
    # where every slot outside the share is kept, the weakest kept one goes, the oldest among
    # equals: the depths below are set by hand, the slots are the bank's own
    before = recall.keys[0].copy()
    depth = recall.depth.copy()
    depth[0, :4] = [9.0, 8.5, 8.25, 8.25]
    recall.depth = depth
    told(brain, next(fillers))
    np.testing.assert_array_equal(recall.keys[0, :6], before[[0, 1, 3, 4, 5, 6]])
    np.testing.assert_array_equal(recall.depth[0, :3], [9.0, 8.5, 8.25])


def test_a_read_renews_the_slots_it_attends_toward_one_telling_and_no_further():
    brain = strong(capacity=8, recall_founder=True, recall_recency=0.0)
    recall = brain.recall
    told(brain, AB, AB, *others(4, seed=7))  # a fact told twice: its first slot deep, its second not
    before = recall.depth[0].copy()
    assert before[0] > 1.8 and before[1] < 1.0
    told(brain, A)  # the name alone, at cosine .71 to the fact: no telling, and a read of it
    p = recall.attended_heads[0][0]  # the attention that read gave the six slots it found
    assert p.shape == (6,) and p[0] == p[1] > 0.1
    faded = before * (1.0 - 1.0 / (cd.Recall.HOLD * 8))
    np.testing.assert_allclose(
        recall.depth[0, :6], faded + p * np.maximum(1.0 - faded, 0.0), rtol=1e-12
    )
    assert recall.depth[0, 0] == faded[0]  # past one telling a read adds nothing
    assert faded[1] < recall.depth[0, 1] <= 1.0  # below it, the read's share of the way back


def test_a_slot_the_reads_keep_fetching_is_held_and_an_unread_one_leaves():
    e = np.eye(12)
    fact = e[[0]] + e[[1]] + e[[2]]
    cues = [e[[0]] + e[[1]], e[[1]] + e[[2]], e[[0]] + e[[2]]]  # each a part of the fact
    used, idle = wide(), wide()
    for brain in (used, idle):
        told(brain, fact)
    spans = {"used": [], "idle": []}
    for step, x in enumerate(apart(160, seed=1)):
        told(used, cues[(step // 2) % 3] if step % 2 else x)
        told(idle, x)
        spans["used"].append(bool(held(used.recall, fact, width=12)))
        spans["idle"].append(bool(held(idle.recall, fact, width=12)))
    assert all(spans["used"])
    assert spans["idle"].index(False) == 15 and not any(spans["idle"][15:])
    place = held(used.recall, fact, width=12)[0]
    assert 0.9 < used.recall.depth[0, place] <= 1.0  # renewed, and no deeper than one telling


def test_reads_that_write_nothing_leave_the_depths_as_they_were(tmp_path):
    brain = strong(capacity=8)
    told(brain, A, B, A, *others(8, seed=8))
    recall = brain.recall
    before = saved(brain, tmp_path, "before.npz")
    depth, keys = recall.depth, recall.keys
    brain.stimulus(A)
    brain.predict(A)
    brain.value(A)
    brain.imagine([A, B, A, A])
    assert recall.depth is depth and recall.keys is keys
    after = saved(brain, tmp_path, "after.npz")
    assert sorted(before) == sorted(after)
    for name in before:
        assert np.array_equal(before[name], after[name]), name
    brain.lesson(A, [1])  # a lesson reads the bank and writes no slot
    assert recall.depth is depth and recall.slots == 8


def test_a_surprising_moment_is_written_deep_and_kept_at_once():
    """With ``surprise=True`` a slot is written at one telling and the brain's surprise at its
    moment: a struck input, over nine times what its neuron tolerates, is kept at once."""
    brain = wide(capacity=8, surprise=True)
    recall = brain.recall
    calm = apart(60, seed=9)  # they never drive the first input
    told(brain, *calm[:30])  # the brain's usual sizes settle
    quiet = float(np.expm1(brain.surprise.moment[0]))
    assert recall.depth[0, -1] == pytest.approx(1.0 + quiet) and quiet < 1.0
    struck = calm[40].copy()
    struck[0, 0] = 80.0
    told(brain, struck)
    tellings = float(np.expm1(brain.surprise.moment[0]))
    assert tellings > 8.0 and recall.depth[0, -1] == pytest.approx(1.0 + tellings)
    told(brain, *calm[30:40], *calm[41:])
    place = held(recall, struck, width=12)
    assert len(place) == 1 and place[0] < 4  # held, outside the protected share
    assert recall.depth[0, place[0]] == pytest.approx(1.0 + tellings)  # a kept depth stays whole
    # the same life without the brain's surprise: the moment is one telling, and it leaves
    plain = wide(capacity=8)
    told(plain, *calm[:30], struck, *calm[30:40], *calm[41:])
    assert not held(plain.recall, struck, width=12)


def test_streams_keep_their_own_depths_and_a_stream_that_begins_holds_none():
    brain = strong(capacity=8, recall_founder=True)
    recall = brain.recall
    left, right = [A, A, *others(10, seed=11)], [*others(10, seed=12), B, B]
    for x, y in zip(left, right, strict=True):
        brain.act(np.concatenate([x, y]), greedy=True)
    keys, depth = law(left, 8, 4, recency=recall.recency)
    np.testing.assert_array_equal(recall.keys[0], keys)
    np.testing.assert_allclose(recall.depth[0], depth, rtol=1e-9)
    keys, depth = law(right, 8, 4, recency=recall.recency)
    np.testing.assert_array_equal(recall.keys[1], keys)
    np.testing.assert_allclose(recall.depth[1], depth, rtol=1e-9)
    assert held(recall, A, 0) == [0] and held(recall, B, 1) == [6, 7]
    kept = recall.depth[0].copy()
    recall.reset(2, rows=np.array([1]))  # the second stream begins again
    assert not recall.held[1].any() and recall.held[0].all()
    np.testing.assert_array_equal(recall.depth[0], kept)
    again = [*others(3, seed=13)]
    for x, y in zip(others(3, seed=14), again, strict=True):
        brain.act(np.concatenate([x, y]), greedy=True)
    np.testing.assert_array_equal(recall.held[1], [False] * 5 + [True] * 3)
    np.testing.assert_allclose(
        recall.depth[1, -3:], law(again, 8, 4, recency=recall.recency)[1], rtol=1e-9
    )
    recall.keep(np.array([1]))
    assert recall.depth.shape == (1, 8) and recall.keys.shape[0] == 1


def test_an_episode_s_end_and_a_boundary_follow_the_law():
    brain = strong(capacity=8)  # a new brain's memory: a boundary slot, the anchor, the grid
    recall = brain.recall
    assert recall.boundary and recall.anchor and recall.relative and recall.grid
    first = [A, A, *others(7, seed=15)]
    told(brain, *first)
    keys, depth = law(first, 8, 4, boundary=True)
    np.testing.assert_array_equal(recall.keys[0], keys)
    np.testing.assert_allclose(recall.depth[0], depth, rtol=1e-9)
    deep = held(recall, A)[0]
    assert recall.depth[0, deep] > 1.7 and recall.slots == 8
    brain.reset(keep=("slots",))  # the episode ends, the slots stay: a boundary joins a full bank
    assert recall.slots == 8 and held(recall, A)[0] <= deep
    assert not recall.keys[0, -1].any() and recall.depth[0, -1] == 1.0  # the boundary, a telling
    assert recall.told[0] == 0
    told(brain, *others(3, seed=16))
    # within the protected share a slot's age is its time: its place in the episode is exact,
    # and a slot older than the episode reads as written before it
    np.testing.assert_array_equal(
        recall.anchored_buckets()[0], [recall.position_buckets] * 5 + [0, 1, 2]
    )
    assert held(recall, A) and recall.depth[0, held(recall, A)[0]] > 1.5


def marked(x: np.ndarray, mark: float) -> np.ndarray:
    """``x`` with a small mark on its last input: a key of its own at cosine .95 or above."""
    out = x.copy()
    out[0, -1] = mark
    return out


def test_the_focus_follows_its_slot_wherever_a_slot_was_given_up():
    """The relative read's focus is the age of the slot the previous read attended most: after
    a write that gives up a slot from the middle it is that slot's age as the bank holds it."""
    brain = strong(capacity=8, recall_founder=True, recall_recency=0.0, recall_relative=True)
    recall = brain.recall
    rng = np.random.default_rng(8)
    alphabet = [A, B, C, AB, *others(4, seed=17)]
    older = middle = 0
    for step in range(120):
        x = marked(alphabet[int(rng.integers(len(alphabet)))], 0.001 * (step + 1))
        before = recall.keys[0].copy() if recall.slots else np.zeros((0, WIDTH))
        told(brain, x)
        read = recall.attended_heads
        if read is None or not len(before):
            continue
        read_at = int(np.argmax(read[0][0]))
        # every key is its moment's alone: where the slot that read attended most lies
        place = np.flatnonzero((recall.keys[0] == before[read_at]).all(axis=1))
        shifted = len(before) < 8 or np.array_equal(before[1:], recall.keys[0, :-1])
        if len(place):
            assert recall.focus[0] == int(recall.ages()[place[0]])
        elif shifted:  # it was the oldest and left: one moment past the bank, as without the law
            assert read_at == 0 and recall.focus[0] == 8
        else:  # it left the middle: no focus
            assert recall.focus[0] == -1
        if not shifted:
            middle += 1
            gone = int(np.flatnonzero((before[:7] != recall.keys[0, :7]).any(axis=1))[0])
            older += int(len(place) and place[0] < gone)
    assert middle > 10 and older > 0  # slots left the middle, some of them younger than the focus
    # a focus on the slot that is given up is none: the fact below is read by its name and,
    # at no depth, is the weakest
    brain = strong(capacity=8, recall_founder=True, recall_recency=0.0, recall_relative=True)
    recall = brain.recall
    told(brain, *others(2, seed=18), AB, *others(5, seed=19))
    assert held(recall, AB) == [2]
    depth = recall.depth.copy()
    depth[0, 2] = 0.0
    recall.depth = depth
    told(brain, A)
    assert int(np.argmax(recall.attended_heads[0][0])) == 2 and not held(recall, AB)
    assert recall.focus[0] == -1


# -- refused attempts and imagination


def full_bank():
    brain = strong(capacity=8)
    told(brain, A, A, B, *others(8, seed=19), A)
    return brain


def test_a_snapshot_is_not_written_into_and_a_restore_returns_to_it():
    brain = full_bank()
    recall = brain.recall
    taken = recall.snapshot()
    assert "depth" in taken and taken["depth"] is recall.depth
    before = {name: np.array(value) for name, value in taken.items()}
    writes = recall.writes
    told(brain, B, *others(5, seed=20), A, B)  # drops from the middle among them
    assert recall.writes == writes + 8 and recall.depth is not taken["depth"]
    for name, value in before.items():
        np.testing.assert_array_equal(taken[name], value, err_msg=name)
    recall.restore(taken)
    for name, value in taken.items():
        assert getattr(recall, name) is value, name
    told(brain, *others(3, seed=21))  # a write after a restore leaves the restored arrays alone
    for name, value in before.items():
        np.testing.assert_array_equal(taken[name], value, err_msg=name)


def test_a_refused_outcome_leaves_the_bank_as_it_was(monkeypatch):
    brain = strong(capacity=8)
    for _ in range(6):
        brain.act(np.concatenate([A, B]))
        brain.learn(np.zeros(2), np.array([False, False]), np.concatenate([A, B]))
    for x in others(4, seed=22):
        brain.act(np.concatenate([x, x]))
        brain.learn(np.zeros(2), np.array([False, False]), np.concatenate([x, x]))
    brain.act(np.concatenate([A, B]))
    recall = brain.recall
    assert recall.slots == 8 and recall.depth.max() > 1.5
    names = (*recall._STATE, "depth")
    before = {name: getattr(recall, name) for name in names}
    copies = {name: np.array(value) for name, value in before.items()}

    def refused(*args, **kwargs):
        raise RuntimeError("the feedback update refused after the memories were touched")

    monkeypatch.setattr(brain.basal_ganglia, "learn", refused)
    with pytest.raises(RuntimeError, match="refused"):
        brain.learn(np.zeros(2), np.array([True, False]), np.concatenate([A, B]))
    for name in names:
        assert getattr(recall, name) is before[name], name
        np.testing.assert_array_equal(getattr(recall, name), copies[name], err_msg=name)


def test_a_refused_settle_leaves_the_bank_as_it_was(tmp_path):
    brain = cd.Brain.compose(WIDTH, 3, seed=8, backend="cpu", recall_strength=True,
                             recall_capacity=8)
    told(brain, A, A, *others(8, seed=23))
    recall = brain.recall
    before = saved(brain, tmp_path, "before.npz")
    depth, keys, writes = recall.depth, recall.keys, recall.writes
    brain.retune(learning_free_steps=0)
    with pytest.raises(RuntimeError, match="did not settle"):
        brain.act(A, greedy=True)
    assert recall.depth is depth and recall.keys is keys and recall.writes == writes
    brain.retune(learning_free_steps=1024)
    after = saved(brain, tmp_path, "after.npz")
    for name in before:
        if name != "brain":
            assert np.array_equal(before[name], after[name]), name
    told(brain, A)
    assert recall.writes == writes + 1 and recall.depth is not depth


def test_imagination_reads_a_full_bank_and_leaves_it_as_it_was(tmp_path):
    brain = full_bank()
    recall = brain.recall
    before = saved(brain, tmp_path, "before.npz")
    state = {name: getattr(recall, name) for name in (*recall._STATE, "depth")}
    phases = brain.imagine([A, B, A, *others(10, seed=24), A])
    assert len(phases) == 14
    for name, value in state.items():
        assert getattr(recall, name) is value, name
    after = saved(brain, tmp_path, "after.npz")
    assert sorted(before) == sorted(after)
    for name in before:
        assert np.array_equal(before[name], after[name]), name
    twin = copy.deepcopy(brain)
    told(twin, A, *others(3, seed=25))
    for name, value in state.items():  # a copy's life is its own
        assert getattr(recall, name) is value, name
    assert not np.shares_memory(twin.recall.depth, recall.depth)


# -- saved continuation


def test_a_saved_brain_continues_with_the_same_slots_and_depths(tmp_path):
    brain = cd.Brain.compose(WIDTH, 3, seed=5, backend="cpu", arousal=True, surprise=True,
                             recall_strength=True, recall_capacity=8, recall_again=0.9,
                             recall_kept=6.0)
    life(brain, 30)
    told(brain, A, A, A)
    recall = brain.recall
    assert recall.slots == 8 and recall.depth.max() > 2.0
    genes = brain.describe()["genes"]
    assert (genes["recall_strength"], genes["recall_again"], genes["recall_kept"]) == (
        True, 0.9, 6.0)
    path = brain.save(tmp_path / "deep.npz")
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive["brain"]))
        np.testing.assert_array_equal(archive["memories/recall/depth"], recall.depth)
    assert meta["format"] == "cadence-brain/5"
    assert meta["features"] == ["recall_strength", "surprise"]
    assert meta["memories"]["recall"]["strength"] is True
    assert (meta["memories"]["recall"]["again"], meta["memories"]["recall"]["kept"]) == (0.9, 6.0)
    twin = cd.Brain.load(path)
    assert twin.recall.strength and twin.describe() == brain.describe()
    np.testing.assert_array_equal(twin.recall.depth, recall.depth)
    rng = np.random.default_rng(6)
    reward = None
    for moment in range(30):
        x = [A, B, (rng.random((1, WIDTH)) < 0.4).astype(float)][moment % 3]
        ours, theirs = brain.live(x, reward=reward), twin.live(x, reward=reward)
        np.testing.assert_array_equal(ours, theirs)
        np.testing.assert_array_equal(brain.recall.keys, twin.recall.keys)
        np.testing.assert_array_equal(brain.recall.depth, twin.recall.depth)
        reward = np.array([float(ours[0] == 0)])
    left, right = saved(brain, tmp_path, "ours.npz"), saved(twin, tmp_path, "theirs.npz")
    assert sorted(left) == sorted(right)
    for name in left:
        assert np.array_equal(left[name], right[name]), name


@pytest.mark.parametrize("tamper", [
    "no depth", "depth of another shape", "negative depth", "depth that is not finite",
    "depth without the option", "the option without its feature", "the feature without the option",
    "kept below zero", "again above one",
])
def test_a_file_whose_depths_do_not_fit_is_refused(tmp_path, tamper):
    brain = strong(capacity=8)
    told(brain, A, A, B)
    arrays = flat(saved(brain, tmp_path, "deep.npz"))
    meta = metadata(arrays)
    assert meta["features"] == ["recall_strength"] and arrays["recall/depth"].shape == (1, 4)
    if tamper == "no depth":
        del arrays["recall/depth"]
    elif tamper == "depth of another shape":
        arrays["recall/depth"] = np.ones((1, 3))
    elif tamper == "negative depth":
        arrays["recall/depth"] = -np.ones((1, 4))
    elif tamper == "depth that is not finite":
        arrays["recall/depth"] = np.full((1, 4), np.inf)
    elif tamper == "depth without the option":
        del meta["recall"]["strength"]
        del meta["features"]
        meta["format"] = "cadence-brain/1"
    elif tamper == "the option without its feature":
        del meta["features"]
        meta["format"] = "cadence-brain/1"
    elif tamper == "the feature without the option":
        del meta["recall"]["strength"]
        del arrays["recall/depth"]
    elif tamper == "kept below zero":
        meta["recall"]["kept"] = -1.0
    elif tamper == "again above one":
        meta["recall"]["again"] = 1.5
    arrays["generic"] = np.array(json.dumps(meta))
    write(tmp_path / "bad.npz", arrays)
    with pytest.raises(ValueError):
        cd.Brain.load(tmp_path / "bad.npz")


def test_a_reader_without_the_feature_refuses_the_file_by_its_name(tmp_path, monkeypatch):
    from cadence import generic

    brain = strong(capacity=8)
    told(brain, A, A)
    path = brain.save(tmp_path / "deep.npz")
    assert cd.Brain.load(path).recall.strength
    monkeypatch.setattr(generic, "FEATURES", frozenset({"strength", "heads"}))
    with pytest.raises(ValueError, match="recall_strength"):
        cd.Brain.load(path)


# -- what the law does not hold with, refused by name


@pytest.mark.parametrize(("options", "match"), [
    ({"recall_strength": 1}, "strength must be boolean"),
    ({"recall_strength": True, "recall_heads": 2}, "strength and heads"),
    ({"recall_strength": True, "recall_window": 2}, "strength and window"),
    ({"recall_strength": True, "recall_action": True}, "strength and action"),
    ({"recall_strength": True, "recall_when": True}, "strength and when"),
    ({"recall_again": 0.9}, "need strength"),
    ({"recall_kept": 4.0}, "need strength"),
    ({"recall_strength": True, "recall_again": 1.5}, "again"),
    ({"recall_strength": True, "recall_again": True}, "again"),
    ({"recall_strength": True, "recall_kept": 0.0}, "kept must be positive"),
    ({"recall_strength": True, "recall_kept": float("inf")}, "kept"),
    ({"recall": False, "recall_strength": True}, "need recall=True"),
])
def test_compose_refuses_what_the_law_does_not_hold_with(options, match):
    with pytest.raises(ValueError, match=match):
        compose(**options)


def test_the_law_s_numbers_retune_and_the_option_is_a_construction_setting():
    brain = strong(capacity=8)
    brain.retune(recall_again=0.8, recall_kept=4.0)
    assert (brain.recall.again, brain.recall.kept) == (0.8, 4.0)
    assert brain.describe()["genes"]["recall_kept"] == 4.0
    told(brain, *[AB] * 5)  # told past the retuned kept depth
    assert brain.recall.depth[0, held(brain.recall, AB)[0]] >= 4.0
    with pytest.raises(ValueError, match="construction setting"):
        brain.retune(recall_strength=False)
    plain = compose()
    with pytest.raises(ValueError, match="construction setting"):
        plain.retune(recall_strength=True)
    for genes in ({"recall_again": 0.8}, {"recall_kept": 4.0}):
        with pytest.raises(ValueError, match="need strength"):
            plain.retune(**genes)
    assert (plain.recall.again, plain.recall.kept) == (0.95, 8.0)
    with pytest.raises(ValueError, match="again"):
        brain.retune(recall_again=-0.1)
    assert brain.recall.again == 0.8


def test_a_memory_refuses_a_surprise_it_cannot_take():
    brain, plain = strong(capacity=8, recall_founder=True), compose(recall_founder=True)
    for subject in (brain, plain):
        told(subject, A)
    state = brain.basal_ganglia.state
    recall = brain.recall
    depth, writes = recall.depth, recall.writes
    for surprise in (np.array([-1.0]), np.array([np.nan]), np.zeros(2), np.array([[1.0]])):
        with pytest.raises(ValueError, match="surprise"):
            recall.witness(B, state, surprise=surprise)
    assert recall.depth is depth and recall.writes == writes
    with pytest.raises(ValueError, match="strength"):
        plain.recall.witness(B, plain.basal_ganglia.state, surprise=np.array([1.0]))
    recall.witness(B, state, surprise=np.array([2.5]))
    assert recall.depth[0, -1] == 3.5


def test_lives_refuse_the_law_by_name_and_the_bank_stays_on_the_host():
    with pytest.raises(ValueError, match="strength"):
        cd.Lives(WIDTH, 3, lives=2, recall_strength=True)
    brain = strong(capacity=8)
    assert not brain.recall.resides and brain._memories_stay_on_host()
    assert compose().recall.resides


def test_the_compiled_moment_takes_the_moments_in_turn_and_the_law_holds():
    stream = [A, A, *others(8, seed=26), A, B, *others(6, seed=27)]
    fast = strong(capacity=8, recall_founder=True, learning_compiled_moment=True)
    slow = strong(capacity=8, recall_founder=True)
    for x in stream:
        a, b = fast.act(x, greedy=True), slow.act(x, greedy=True)
        np.testing.assert_array_equal(a, b)
    keys, depth = law(stream, 8, 4, recency=fast.recall.recency)
    for brain in (fast, slow):
        np.testing.assert_array_equal(brain.recall.keys[0], keys)
        np.testing.assert_allclose(brain.recall.depth[0], depth, rtol=1e-9)
    assert held(fast.recall, A)[0] == 0 and len(stream) > 2 * 8


def test_a_brain_on_a_torch_device_keeps_the_bank_on_the_host():
    pytest.importorskip("torch")
    stream = [A, A, *others(8, seed=28), A, B]
    brain = cd.Brain.compose(WIDTH, 3, modules=(8,), seed=3, backend="torch", device="cpu",
                             precision="float64", episodic=False, recall_strength=True,
                             recall_capacity=8, recall_founder=True)
    for x in stream:
        brain.act(x, greedy=True)
    recall = brain.recall
    assert brain.brain.backend == "torch" and recall._device is None
    keys, depth = law(stream, 8, 4, recency=recall.recency)
    np.testing.assert_array_equal(recall.keys[0], keys)
    np.testing.assert_allclose(recall.depth[0], depth, rtol=1e-6)


def test_a_grown_brain_with_a_slots_strength_names_it_in_its_file(tmp_path):
    """grow adds a slot's strength to a brain without a recall memory: the grown brain saves
    cadence-brain/5 with the feature, and loads."""
    import json

    import cadence as cd

    brain = cd.Brain.compose(6, 3, seed=0, backend="cpu", recall=False, arousal=True)
    brain.live(np.ones((1, 6)))
    grown = brain.grow(recall=True, recall_strength=True)
    grown.live(np.ones((1, 6)))
    path = grown.save(tmp_path / "grown.npz")
    with np.load(path, allow_pickle=False) as archive:
        saved = json.loads(str(archive["brain"]))
    assert saved["format"] == "cadence-brain/5" and saved["features"] == ["recall_strength"]
    assert cd.Brain.load(path).recall.strength


def test_attention_and_recall_strength_are_refused_with_the_window_named():
    with pytest.raises(ValueError, match="recall_strength does not compose with attention=True"
                       ) as refused:
        cd.Brain.compose(53, 4, learned_code=64, attention=True, recall_strength=True, seed=0)
    assert "recall_window=2" in str(refused.value)
