"""The recall memory's position term: every head reads each slot's age, as the address reads the
moment, and lessons teach it.

The founder builds it (32 ages with a bucket each), zero at birth, so a brain reads as one
without it until a lesson moves it; ``recall_position=0`` builds none; a checkpoint saved before
the term loads without one, and a two-head file of 0.89.0, the release that added it, is refused.
A moment chooses how far back its head looks: the term is read from the event heard and the
cortex of the moment before.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from checkpoints import flat, write

import cadence as cd
from cadence.stream import _share

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "recall_0890"


def composed(**options):
    return cd.Brain.compose(8, 3, modules=(16,), seed=3, episodic=False, recall=True, **options)


def stream(n=12, seed=0):
    rng = np.random.default_rng(seed)
    return [np.eye(8)[[int(rng.integers(8))]] for _ in range(n)]


def test_the_founder_reads_each_slots_age_and_zero_builds_none():
    founder = composed()
    recall = founder.recall
    assert recall.position == 32 == cd.Recall.options()["position"]
    assert recall.position_rate == 0.1
    assert recall.position_buckets == cd.Recall.position_count(32, recall.capacity) == 33
    assert len(recall.position_maps) == recall.heads
    p_x, p_c, p = recall.position_maps[0]
    assert p_x.shape == (recall.key_width, 33) and p_c.shape == (len(recall.association), 33)
    assert not p_x.any() and not p_c.any() and not p.any()
    # the founder is left out of the description; a memory without the term names it
    assert "recall_position" not in founder.describe()["genes"]
    off = composed(recall_position=0)
    assert off.recall.position_maps == [] and off.recall.position_buckets == 0
    assert off.describe()["genes"]["recall_position"] == 0


@pytest.mark.parametrize(("position", "capacity", "count"), [
    (32, 64, 33), (32, 320, 36), (32, 32, 32), (8, 4, 4), (0, 64, 0), (16, 17, 17),
])
def test_the_buckets_are_exact_then_one_per_doubling(position, capacity, count):
    assert cd.Recall.position_count(position, capacity) == count


def test_each_slot_has_the_bucket_of_its_age():
    brain = composed(recall_position=4, recall_capacity=40)
    for x in stream(37):
        brain.act(x, greedy=True)
    recall = brain.recall
    ages = recall.ages()
    expected = [int(a) if a < 4 else 4 + int(np.floor(np.log2(a / 4))) for a in ages]
    assert recall.buckets().tolist() == [min(b, recall.position_buckets - 1) for b in expected]
    assert recall.buckets()[-1] == 0 and recall.buckets()[0] == recall.position_buckets - 1


def test_a_brain_reads_as_without_the_term_until_a_lesson():
    on, off = composed(recall_heads=2), composed(recall_heads=2, recall_position=0)
    for x in stream(16):
        on.act(x, greedy=True)
        off.act(x, greedy=True)
        a, b = on.recall.attended_heads, off.recall.attended_heads
        assert (a is None) == (b is None)
        if a is not None:
            for p, q in zip(a, b, strict=True):
                np.testing.assert_array_equal(p, q)


@pytest.mark.filterwarnings("ignore:lesson on the observations")
def test_lessons_teach_the_terms_whatever_the_address_rate():
    brain = composed(recall_address=0.0)
    xs = stream(24, seed=1)
    for x in xs[:6]:
        brain.act(x, greedy=True)
    for k, x in enumerate(xs[6:]):
        brain.lesson(x, np.array([k % 3]))
        brain.act(x, greedy=True)
    assert brain.last_learning["demonstration_position_rows"] >= 1.0
    assert brain.last_learning["demonstration_address_rows"] == 0.0  # the address map is frozen
    assert any(np.abs(m).sum() > 0 for maps in brain.recall.position_maps for m in maps)
    still = composed(recall_position_rate=0.0)
    for k, x in enumerate(xs):
        still.lesson(x, np.array([k % 3]))
        still.act(x, greedy=True)
    assert not any(np.abs(m).sum() for maps in still.recall.position_maps for m in maps)


def test_a_moment_chooses_how_far_back_its_head_looks():
    """Two slots of one event, written ten moments apart, and fillers between: with the terms
    set so that a cue raises the old bucket and its absence the recent one, the read follows
    the cue."""
    brain = composed(recall_address=0.0, recall_temperature=0.25)
    same, cue = np.eye(8)[[1]], np.eye(8)[[7]]
    fillers = [np.eye(8)[[k]] for k in (2, 3, 4, 5, 6)]
    for x in [same, *fillers, *fillers[:4], same, *fillers[:2]]:
        brain.act(x, greedy=True)
    recall = brain.recall
    ages = recall.ages()
    events = recall.values[0].argmax(axis=1)
    old, recent = [int(i) for i in np.flatnonzero(events == 1)]
    b_old, b_recent = recall.buckets()[old], recall.buckets()[recent]
    p_x, p_c, p = (a.copy() for a in recall.position_maps[0])
    p_x[7, b_old] += 6.0  # hearing the cue looks back to the old slot's age
    p[b_recent] += 3.0  # otherwise the recent one is preferred
    recall.position_maps = [(p_x, p_c, p)]
    assert ages[old] > ages[recent]
    for query, chosen in ((np.maximum(same, cue), old), (same, recent)):
        snapshot = recall.snapshot()
        brain.act(query, greedy=True)
        attended = recall.attended_heads[0][0]
        assert int(np.argmax(attended)) == chosen
        recall.restore(snapshot)


@pytest.mark.filterwarnings("ignore:lesson on the observations")
def test_the_terms_survive_a_checkpoint_and_the_rate_retunes(tmp_path):
    brain = composed(recall_heads=2)
    xs = stream(20, seed=2)
    for k, x in enumerate(xs):
        brain.lesson(x, np.array([k % 3]))
        brain.act(x, greedy=True)
    loaded = cd.Brain.load(brain.save(tmp_path / "taught.npz"))
    assert loaded.recall.position == 32
    for kept, saved in zip(loaded.recall.position_maps, brain.recall.position_maps, strict=True):
        for a, b in zip(kept, saved, strict=True):
            np.testing.assert_array_equal(a, b)
    assert loaded.describe() == brain.describe()
    loaded.retune(recall_position_rate=0.05)
    assert loaded.recall.position_rate == 0.05
    with pytest.raises(ValueError):
        loaded.retune(recall_position=8)


@pytest.mark.filterwarnings("ignore:lesson on the observations")
def test_a_checkpoint_saved_before_the_term_loads_without_one(tmp_path):
    """A file of an earlier version names no position option and holds no position maps: it
    loads without the term and reads as it did."""
    brain = composed(recall_position=0)
    for k, x in enumerate(stream(14, seed=4)):
        brain.lesson(x, np.array([k % 3]))
        brain.act(x, greedy=True)
    path = brain.save(tmp_path / "saved.npz")
    with np.load(path, allow_pickle=False) as data:
        arrays = flat({name: data[name].copy() for name in data.files})
    meta = json.loads(str(arrays["generic"]))
    assert meta["recall"].pop("position") == 0  # as an earlier version wrote it: no option
    assert not any("position" in name for name in arrays)
    arrays["generic"] = np.array(json.dumps(meta))
    write(tmp_path / "earlier.npz", arrays)
    loaded = cd.Brain.load(tmp_path / "earlier.npz")
    assert loaded.recall.position == 0 and loaded.recall.position_maps == []
    x = stream(1, seed=9)[0]
    np.testing.assert_array_equal(loaded.act(x, greedy=True), cd.Brain.load(path).act(x, greedy=True))


@pytest.mark.filterwarnings("ignore:lesson on the observations")
def test_a_refused_step_leaves_the_maps_as_they_were():
    brain = composed()
    recall = brain.recall
    before = [tuple(a.copy() for a in maps) for maps in recall.position_maps]
    recall.position_rate = float("inf")  # every step overflows
    for k, x in enumerate(stream(10, seed=5)):
        brain.lesson(x, np.array([k % 3]))
        brain.act(x, greedy=True)
    for kept, old in zip(recall.position_maps, before, strict=True):
        for a, b in zip(kept, old, strict=True):
            assert np.isfinite(a).all()
            np.testing.assert_array_equal(a, b)


@pytest.mark.filterwarnings("ignore:lesson on the observations")
def test_every_lesson_reports_the_position_keys_and_parameters_count_the_maps():
    """The report's keys do not depend on whether the memory held slots: a lesson on an empty
    memory reports the position rows as zero, as later lessons report them."""
    brain = composed()
    first = brain.lesson(np.eye(8)[[1]], np.array([1]))
    assert first["position_rows"] == 0.0 and first["position_refused"] == 0.0
    brain.act(np.eye(8)[[1]], greedy=True)
    later = brain.lesson(np.eye(8)[[2]], np.array([2]))
    assert set(first) == set(later)
    off = composed(recall_position=0)
    assert "position_rows" not in off.lesson(np.eye(8)[[1]], np.array([1]))
    sizes = sum(a.size for maps in brain.recall.position_maps for a in maps)
    assert brain.parameters() - off.parameters() == sizes > 0


@pytest.mark.filterwarnings("ignore:lesson on the observations")
def test_the_term_and_the_weights_give_the_attention_the_read_used():
    brain = composed(recall_heads=2)
    xs = stream(30, seed=6)
    for k, x in enumerate(xs[:24]):
        brain.lesson(x, np.array([k % 3]))
        brain.act(x, greedy=True)
    recall = brain.recall
    assert any(np.abs(m).sum() > 0 for maps in recall.position_maps for m in maps)
    brain.stimulus(xs[24])  # the read addresses the moment and keeps its attention
    moment, attended = recall.last, recall.attended_heads
    left = None
    for h in range(recall.heads):
        address = moment.address if h == 0 else recall._head_address(moment, h)
        term = recall.position_term(h, moment.event, moment.cortex)
        p = recall.weights(address, term)
        if left is not None:  # a competing head takes at most what the heads before it left
            logits = recall._attention(address, recall.keys, None, term)
            p = _share(logits, left, p.sum(axis=1))
        np.testing.assert_allclose(p, attended[h])
        left = 1.0 - p if left is None else left * (1.0 - p)
        assert recall.position_map(h) is recall.position_maps[h]
    with pytest.raises(ValueError):
        recall.position_map(2)
    with pytest.raises(ValueError):
        composed(recall_position=0).recall.position_map(0)


def test_the_maps_go_by_their_checkpoint_names():
    """``map_names(kind, head)`` gives the checkpoint names of one head's three maps of a kind,
    head 0's bare and head h's with ``_h``; the head-0 aliases of 0.92.1 (``w_x``, ``w_c``,
    ``b``, ``position_x``, ``position_c``, ``position_b``, ``attended``) are gone, the maps
    being ``head_map(0)``, ``position_map(0)`` and ``attended_heads[0]``."""
    recall = composed().recall
    assert cd.Recall.map_names("address", 0) == ("w_x", "w_c", "b")
    assert cd.Recall.map_names("address", 2) == ("w_x_2", "w_c_2", "b_2")
    assert cd.Recall.map_names("position", 0) == ("position_x", "position_c", "position_b")
    assert cd.Recall.map_names("anchor", 1) == ("anchor_x_1", "anchor_c_1", "anchor_b_1")
    assert cd.Recall.map_names("relative", 0)[2] == "relative_b"
    assert cd.Recall.map_names("slope", 3)[0] == "slope_x_3"
    with pytest.raises(ValueError, match="unknown map kind"):
        cd.Recall.map_names("lean", 0)
    saved = recall.arrays()
    assert set(cd.Recall.map_names("address", 0)) | set(cd.Recall.map_names("position", 0)) <= set(saved)
    for name, kept in zip(cd.Recall.map_names("address", 0), recall.head_map(0), strict=True):
        assert saved[name] is kept
    for name in ("w_x", "w_c", "b", "position_x", "position_c", "position_b", "attended"):
        assert not hasattr(recall, name)
    two = composed(recall_heads=2).recall
    assert set(cd.Recall.map_names("address", 1)) <= set(two.arrays())
    assert len(two.head_maps) == 2 and two.head_map(1) is two.head_maps[1]


def test_a_two_head_brain_saved_by_0_89_0_is_refused_by_name():
    """A two-head recall brain saved by cadence-net 0.89.0, the fixture: its heads read alike, and
    cadence 0.94.0 runs it."""
    with pytest.raises(ValueError, match="heads read alike.*cadence 0.94.0 takes it"):
        cd.Brain.load(FIXTURE / "brain.npz")
