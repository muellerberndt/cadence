"""Several heads over one bank, by what every read holds (``tests/recall_watch.py``): a slot is
read in full once per moment, so the heads' shares of a slot sum to at most one; a later head
takes its whole mass while enough is left, by the water-filling of its own attention; each port
holds its head's attention over the slots; and a head's lesson takes the attention the head read
with. Checked at every read of lives of each kind of moment, on each backend, beside other
streams, after a save and after heads were grown, with the bank compared slot by slot against a
list per stream, and in the long-term store's heads."""

from __future__ import annotations

import copy

import numpy as np
import pytest
from places import PLACES, place
from recall_watch import reference_share, watch

import cadence as cd
from cadence.longterm import LongTermMemory
from cadence.stream import _left, _share

INPUTS, ACTIONS = 12, 4
EVENTS = np.eye(INPUTS)
COMPOSITIONS = {
    "attention recipe": dict(learned_code=16, attention=True),
    "composed recall": dict(recall=True),
    "founder recall": dict(recall=True, recall_founder=True, recall_address=0.01, episodic=False),
    "rewarded maps": dict(recall=True, recall_address_reward=1.0),
}


def composed(name: str, heads: int, seed: int = 0, **options) -> cd.Brain:
    settings = {"backend": "cpu", **COMPOSITIONS[name], **options}
    return cd.Brain.compose(INPUTS, ACTIONS, modules=(16,), seed=seed, recall_heads=heads,
                            **settings)


def moments(rng: np.random.Generator, streams: int, count: int) -> list[np.ndarray]:
    return [
        EVENTS[rng.integers(0, INPUTS, size=streams)] + 0.05 * rng.random((streams, INPUTS))
        for _ in range(count)
    ]


def lived(brain: cd.Brain, rng: np.random.Generator, streams: int, steps: int = 12) -> None:
    """Lessons with a greedy act after each, then steps with rewards, a teacher at some of them
    and episodes that end on some streams."""
    for x in moments(rng, streams, 3):
        brain.lesson(x, rng.integers(0, ACTIONS, size=streams))
        brain.act(x, greedy=True)
    brain.step(moments(rng, streams, 1)[0])
    for t, x in enumerate(moments(rng, streams, steps)):
        outcome = {"reward": rng.random(streams) - 0.3}
        if t % 3 == 1:
            outcome["teacher"] = rng.integers(0, ACTIONS, size=streams)
        if t % 5 == 4:
            outcome["done"] = rng.random(streams) < 0.5
        brain.step(x, **outcome)


# -- the shares of the heads in turn


def test_a_later_heads_shares_are_the_water_filling_of_what_is_left():
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(800):
        n = int(rng.integers(1, 12))
        logits = rng.normal(size=n) * float(rng.choice([0.1, 1.0, 5.0, 40.0, 300.0]))
        left = [
            rng.random(n),
            np.where(rng.random(n) < 0.4, 0.0, rng.random(n)),
            np.where(rng.random(n) < 0.5, 1.0, rng.random(n) * 1e-9),
            np.ones(n),
            np.clip(1.0 - rng.dirichlet(np.ones(n) * 0.3), 0.0, 1.0),
        ][int(rng.integers(5))]
        if rng.random() < 0.2:
            logits[rng.integers(0, n)] = -np.inf
        mass = float(rng.choice([1.0, 1.0, rng.random(), 0.0]))
        shares = _share(logits[None, :], left[None, :], np.array([mass]))[0]
        assert np.isfinite(shares).all() and (shares >= 0).all() and (shares <= left + 1e-15).all()
        worst = max(worst, float(np.abs(shares - reference_share(logits, left, mass)).max()))
    assert worst < 1e-9


@pytest.mark.parametrize("heads", [2, 3, 4, 7])
def test_heads_in_turn_read_no_slot_above_one_and_each_takes_what_is_due(heads):
    rng = np.random.default_rng(heads)
    for _ in range(300):
        n = int(rng.integers(1, 10))
        scale = float(rng.choice([0.2, 1.0, 4.0, 30.0]))
        logits = [rng.normal(size=(1, n)) * scale for _ in range(heads)]
        p = np.exp(logits[0] - logits[0].max())
        p /= p.sum()
        reads, left = [p], _left(p)
        for h in range(1, heads):
            p = _share(logits[h], left, np.ones(1))
            assert p.sum() == pytest.approx(min(1.0, float(left.sum())), abs=1e-9)
            reads.append(p)
            left = _left(p, left)
        total = np.sum(reads, axis=0)
        assert total.max() <= 1.0 + 1e-12
        assert total.sum() == pytest.approx(min(float(heads), float(n)), abs=1e-9)


def test_a_slot_read_whole_leaves_exactly_nothing_to_the_heads_after():
    """Rounding leaves about 1e-16 of a slot the heads took whole; no later head reads it."""
    rng = np.random.default_rng(0)
    for _ in range(4000):
        n, heads = int(rng.integers(1, 6)), int(rng.integers(3, 6))
        tied = rng.random() < 0.5
        logits = [np.zeros((1, n)) if tied else rng.normal(size=(1, n)) for _ in range(heads)]
        p = np.exp(logits[0] - logits[0].max())
        p /= p.sum()
        left = _left(p)
        for h in range(1, heads):
            p = _share(logits[h], left, np.ones(1))
            assert p.sum() == 0.0 or p.sum() > 1e-9
            left = _left(p, left)


# -- every read of a life


@pytest.mark.parametrize("streams", [1, 3])
@pytest.mark.parametrize("heads", [3, 4])
@pytest.mark.parametrize("name", COMPOSITIONS)
def test_every_read_of_a_life_holds_with_several_heads(name, heads, streams):
    brain = composed(name, heads, seed=heads + streams, recall_capacity=5)
    seen = watch(brain)
    lived(brain, np.random.default_rng(100 + heads), streams)
    assert seen.broken == []
    assert seen.reads >= 30 and seen.lessons >= heads
    assert seen.largest_total <= 1.0 + 1e-9
    assert seen.heads_reading == min(heads, 5)  # every head read, the bank's slots allowing


def test_six_heads_over_five_slots_leave_the_last_head_nothing():
    brain = composed("founder recall", 6, recall_capacity=5)
    seen = watch(brain)
    lived(brain, np.random.default_rng(6), 1)
    assert seen.broken == [] and seen.heads_reading == 5
    assert seen.largest_total == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("name", ["attention recipe", "composed recall"])
def test_every_read_holds_through_live_and_wait(name):
    brain = composed(name, 3, seed=3, arousal=True, recall_capacity=6)
    seen = watch(brain)
    rng = np.random.default_rng(9)
    brain.live(moments(rng, 1, 1)[0])
    for t, x in enumerate(moments(rng, 1, 24)):
        if t % 5 == 3:
            brain.wait(x)
            continue
        given = {}
        if brain.pending_feedback:
            given["reward"] = [float(rng.random() - 0.4)]
            if t % 7 == 6:
                given["done"] = [True]
        if t % 6 == 2:
            given["lesson"] = (np.eye(ACTIONS)[[int(rng.integers(ACTIONS))]], 1.0)
        brain.live(x, **given)
    assert seen.broken == [] and seen.reads >= 24


@pytest.mark.parametrize("name", PLACES)
def test_every_read_holds_on_each_backend(name):
    where = place(name)
    brain = composed("attention recipe", 3, seed=1, **where)
    seen = watch(brain, tolerance=1e-6)
    rng = np.random.default_rng(5)
    for x in moments(rng, 2, 8):
        brain.lesson(x, rng.integers(0, ACTIONS, size=2))
        brain.act(x, greedy=True)
    assert seen.broken == [] and seen.heads_reading == 3


def test_heads_grown_on_a_trained_brain_read_as_heads_composed():
    brain = composed("founder recall", 2, seed=5)
    rng = np.random.default_rng(2)
    for x in moments(rng, 2, 6):
        brain.lesson(x, rng.integers(0, ACTIONS, size=2))
        brain.act(x, greedy=True)
    before = [tuple(np.array(a) for a in brain.recall.head_map(h)) for h in range(2)]
    grown = brain.grow(recall_heads=4)
    for h in range(2):  # the heads it had keep their maps
        for kept, held in zip(before[h], grown.recall.head_map(h), strict=True):
            np.testing.assert_array_equal(kept, held)
    seen = watch(grown)
    for x in moments(rng, 2, 8):
        grown.lesson(x, rng.integers(0, ACTIONS, size=2))
        grown.act(x, greedy=True)
    assert seen.broken == [] and seen.heads_reading == 4


# -- beside other streams, and after a save


@pytest.mark.parametrize("name", ["attention recipe", "composed recall", "founder recall"])
def test_a_streams_read_is_its_own_beside_other_streams(name):
    brain = composed(name, 4, seed=2, recall_capacity=6)
    rng = np.random.default_rng(3)
    lived(brain, rng, 3)
    # one stream's episode ends: it begins again and holds fewer slots than the others
    brain.step(moments(rng, 3, 1)[0], reward=np.zeros(3), done=np.array([True, False, False]))
    recall = brain.recall
    assert len({int(n) for n in np.asarray(recall.held).sum(axis=1)}) > 1  # unequal banks
    code = rng.random((3, recall.key_width))
    whole = copy.deepcopy(recall)
    whole.stimulate(np.zeros((3, brain.connectome.n)), code)
    together = whole.attended_heads
    compared = 0
    for row in range(3):
        one = copy.deepcopy(recall)
        one.keep(np.array([row]))
        one.stimulate(np.zeros((1, brain.connectome.n)), code[[row]])
        for alone, beside in zip(one.attended_heads, together, strict=True):
            if alone is None or beside is None:
                assert alone is None and beside is None
                continue
            np.testing.assert_allclose(alone[0], beside[row], rtol=0, atol=1e-12)
            compared += 1
    assert compared >= 8


@pytest.mark.parametrize("name", ["attention recipe", "composed recall"])
@pytest.mark.parametrize("heads", [3, 4])
def test_a_saved_brain_of_several_heads_continues_as_the_brain_not_saved(name, heads, tmp_path):
    brain = composed(name, heads, seed=1, recall_capacity=6)
    rng = np.random.default_rng(7)
    lived(brain, rng, 2)  # a sampled action awaits its outcome in the file
    loaded = cd.Brain.load(brain.save(tmp_path / "mid.npz"))
    seen = watch(loaded)
    for x in moments(rng, 2, 6):
        reward = rng.random(2) - 0.3
        np.testing.assert_array_equal(brain.step(x, reward=reward), loaded.step(x, reward=reward))
    assert seen.broken == [] and seen.reads >= 6 and seen.heads_reading == heads
    with np.load(brain.save(tmp_path / "a.npz")) as a, np.load(loaded.save(tmp_path / "b.npz")) as b:
        for member in a.files:
            if a[member].dtype.kind in "fiub":
                np.testing.assert_array_equal(a[member], b[member], err_msg=member)


# -- the bank, slot by slot


@pytest.mark.parametrize("boundary", [False, True], ids=["no boundary", "boundary"])
@pytest.mark.parametrize("capacity", [3, 16])
def test_the_bank_holds_each_streams_moments_in_their_order(boundary, capacity):
    """Against a list per stream: an accepted moment is a slot of its stream, the slot before
    learns it as its next, an ended episode empties its stream to a boundary where the memory
    keeps one, and beyond the capacity the oldest slot leaves."""
    streams, width = 4, 10
    brain = cd.Brain.compose(width, 3, modules=(12,), seed=capacity, backend="cpu", recall=True,
                             recall_founder=True, recall_heads=3, recall_capacity=capacity,
                             recall_boundary=boundary, recall_address=0.01, episodic=False)
    recall, rng, zero = brain.recall, np.random.default_rng(capacity), np.zeros(width)
    start = [[zero, zero]] if boundary else []
    lists = [[list(slot) for slot in start] for _ in range(streams)]

    def written(x: np.ndarray) -> None:
        for row in range(streams):
            if lists[row]:
                lists[row][-1][1] = x[row]
            lists[row].append([x[row], zero])
            del lists[row][: max(len(lists[row]) - capacity, 0)]
            held = np.asarray(recall.held)[row]
            np.testing.assert_array_equal(
                np.asarray(recall.values)[row][held], np.array([s[0] for s in lists[row]])
            )
            np.testing.assert_array_equal(
                np.asarray(recall.nexts)[row][held], np.array([s[1] for s in lists[row]])
            )

    x = rng.random((streams, width))
    brain.step(x)
    written(x)
    for _ in range(24):
        x, done = rng.random((streams, width)), rng.random(streams) < 0.2
        brain.step(x, reward=rng.random(streams) - 0.3, done=done)
        for row in np.flatnonzero(done):
            lists[row] = [list(slot) for slot in start]
        written(x)


# -- the long-term store's heads


def test_the_long_term_stores_heads_read_and_credit_no_record_above_one():
    rng = np.random.default_rng(4)
    for _ in range(200):
        width, count, heads = int(rng.integers(4, 12)), int(rng.integers(1, 9)), int(rng.integers(1, 7))
        memory = LongTermMemory(
            width, 3, 16, candidates=0, temperature=float(rng.choice([0.0, 0.05, 0.3, 1.0, 5.0])),
            cosine=bool(rng.integers(2)),
        )
        for _ in range(count):
            memory.write(rng.normal(size=width), rng.normal(size=3))
        read = memory.read(rng.normal(size=width), heads=heads, credit=False)
        assert all((np.asarray(w) >= 0).all() and np.sum(w) <= 1 + 1e-9 for w in read.weights)
        assert np.sum(read.weights, axis=0).max() <= 1 + 1e-12
        own = memory._weights_each(rng.normal(size=(heads, len(read.slots))))
        assert np.sum(own, axis=0).max() <= 1 + 1e-12
        deep = memory.deep[: memory.count].copy()
        memory.read(rng.normal(size=width), heads=heads)  # a read deepens what its heads read
        assert (memory.deep[: memory.count] - deep).max() <= 1 + 1e-9
