"""The recall memory's position term: every head reads each slot's age, as the address reads the
moment, and lessons teach it.

The founder builds it (32 ages with a bucket each), zero at birth, so a brain reads as one
without it until a lesson moves it; ``recall_position=0`` builds none; a checkpoint saved before
the term loads without one, and one saved by 0.89.0, the release that added it, reads as it did.
A moment chooses how far back its head looks: the term is read from the event heard and the
cortex of the moment before.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from checkpoints import earlier

import cadence as cd
from cadence import brain as cadence_brain
from cadence.stream import _share

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "recall_0890"
# The fixture's continuation was recorded on the NumPy solver; the compiled dense kernel orders
# the arithmetic otherwise, so its actions, attention and maps are read there alone
# (CADENCE_FUSED=0 selects it).
NUMPY_SOLVER = not cadence_brain._FUSED


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
        attended = recall.attended[0]
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
        arrays = earlier({name: data[name].copy() for name in data.files})
    meta = json.loads(str(arrays["generic"]))
    assert meta["recall"].pop("position") == 0  # as an earlier version wrote it: no option
    assert not any("position" in name for name in arrays)
    arrays["generic"] = np.array(json.dumps(meta))
    np.savez(tmp_path / "earlier.npz", **arrays)
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


def test_head_zeros_maps_go_by_their_checkpoint_names():
    recall = composed().recall
    assert recall.position_x is recall.position_maps[0][0]
    assert recall.position_c is recall.position_maps[0][1]
    assert recall.position_b is recall.position_maps[0][2]
    assert set(cd.Recall.position_names(0)) <= set(recall.arrays())
    with pytest.raises(ValueError):
        _ = composed(recall_position=0).recall.position_x


def continued_0890(brain, moments=8):
    """The fixture's continuation: one-hot moments of ``default_rng(11)``, each unlike the moment
    before it (the first unlike the saved life's last, input 1), a lesson of label ``k % 2``
    before a greedy act on each. Returns the actions, every head's attention and the maps."""
    rng = np.random.default_rng(11)
    last, actions, attention = 1, [], []
    for k in range(moments):
        last = int(rng.choice([i for i in range(4) if i != last]))
        x = np.eye(4)[[last]]
        brain.lesson(x, np.array([k % 2]))
        actions.append(int(np.asarray(brain.act(x, greedy=True)).reshape(-1)[0]))
        attention.append([np.asarray(p, dtype=np.float64).reshape(-1)
                          for p in brain.recall.attended_heads])
    maps = [np.asarray(m, dtype=np.float64) for head in brain.recall.position_maps for m in head]
    return actions, attention, maps


def sha256_of(arrays):
    digest = hashlib.sha256()
    for a in arrays:
        digest.update(np.ascontiguousarray(a).tobytes())
    return digest.hexdigest()


def test_a_brain_saved_by_0_89_0_loads_and_reads_as_it_did(tmp_path):
    """A two-head recall brain composed, taught and saved by cadence-net 0.89.0 (the fixture; a
    capacity of 40, so ages from 32 share the doubling bucket) loads at 0.89.0's settings with
    no option of a later release, its heads reading alike; continued with lessons, a save of it
    continues as it does, and on the NumPy solver its actions, both heads' attention and the
    position maps are the ones 0.89.0 recorded."""
    provenance = json.loads((FIXTURE / "provenance.json").read_text())
    brain = cd.Brain.load(FIXTURE / "brain.npz")
    recall = brain.recall
    assert {name: getattr(recall, name) for name in provenance["settings"]} == (
        provenance["settings"]
    )
    assert recall.position_buckets == 33 and not recall.compete
    assert recall.slope_maps == [] and recall.anchor_maps == [] and recall.relative_maps == []
    assert (recall.slope, recall.when, recall.boundary, recall.position_reward) == (
        0.0, False, False, 0.0
    )
    genes = {k: v for k, v in brain.describe()["genes"].items() if k.startswith("recall")}
    assert genes == {**provenance["genes"], "recall_compete": False}
    assert brain.parameters() == provenance["parameters"]
    again = cd.Brain.load(brain.save(tmp_path / "again.npz"))
    actions, attention, maps = continued_0890(brain)
    resaved = continued_0890(again)
    assert resaved[0] == actions
    for kept, read in zip(resaved[1], attention, strict=True):
        for a, b in zip(kept, read, strict=True):
            np.testing.assert_array_equal(a, b)
    for a, b in zip(resaved[2], maps, strict=True):
        np.testing.assert_array_equal(a, b)
    if NUMPY_SOLVER:
        recorded = provenance["continuation"]
        assert actions == recorded["actions"]
        assert sha256_of([p for heads in attention for p in heads]) == recorded["attention_sha256"]
        assert sha256_of(maps) == recorded["position_maps_sha256"]
