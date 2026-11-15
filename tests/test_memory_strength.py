"""A record's strength (cadence#336): surprise and repetition strengthen it, a kept threshold
holds it for good, the rest fades and is forgotten.

``LongTermMemory(blend=, familiar=, keep=, forget=, novelty=)`` and ``write(..., surprise=)``,
and ``SynapticMemory(fade_keep=)``. Every setting is off by default; off, the store is the store
it was (G5)."""

from __future__ import annotations

import numpy as np
import pytest

from cadence.longterm import LongTermMemory
from cadence.memory import SynapticMemory

WIDTH = 32


def unit(x):
    x = np.asarray(x, dtype=float)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def take(rng, clean, spread=0.6):
    """A noisy take of a pattern, at cosine about .86 to it."""
    return unit(clean + spread * rng.standard_normal(len(clean)) / np.sqrt(len(clean)))


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def sensory(capacity=200, **settings):
    """A store of sensory records: read by cosine, a take at cosine .6 or above is its record
    told again, a depth that does not leak over these short lives."""
    return LongTermMemory(WIDTH, 64, capacity, cosine=True, revise=0.6, hold=float("inf"),
                          **settings)


def recalled(store, rng, clean, label):
    """Whether a noisy cue of ``clean`` reads the record whose value is ``label``."""
    read = store.read(take(rng, clean), credit=False)
    return read.first() >= 0 and int(np.argmax(store.value(read))) == label


def test_ten_noisy_takes_settle_on_one_record_nearer_the_pattern_than_any_take():  # G4
    rng = np.random.default_rng(0)
    clean = unit(rng.standard_normal(WIDTH))
    takes = [take(rng, clean) for _ in range(10)]
    blended, plain = sensory(blend=True), sensory()
    for t in takes:
        blended.write(t, np.eye(64)[3])
        plain.write(t, np.eye(64)[3])
    assert blended.count == plain.count == 1 and blended.deep[0] == 10.0
    nearest = max(cosine(t, clean) for t in takes)
    assert cosine(blended.keys[0], clean) > nearest  # the mean of the takes
    np.testing.assert_allclose(blended.keys[0], np.mean(takes, axis=0), atol=1e-12)
    np.testing.assert_array_equal(plain.keys[0], takes[-1])  # without blend: the last take
    with pytest.raises(ValueError, match="cosine"):
        LongTermMemory(WIDTH, 4, 10, blend=True)


def test_a_deeper_record_reads_ahead_of_an_equal_match_with_familiar():
    for familiar, first in ((0.0, 0), (0.01, 1)):
        store = LongTermMemory(WIDTH, 4, 10, cosine=True, revise=0.999, familiar=familiar)
        store.write(np.eye(WIDTH)[0], np.eye(4)[0])  # shallow, told once
        for _ in range(3):  # deep, told three times
            store.write(np.eye(WIDTH)[1], np.eye(4)[1])
        assert store.deep[:2].tolist() == [1.0, 3.0]
        tie = np.eye(WIDTH)[0] + np.eye(WIDTH)[1]  # the same cosine to both
        assert store.read(tie, credit=False).first() == first
        many = store.read_each(np.stack([tie, tie]), credit=False)
        assert many.first(0) == first and many.first(1) == 1 - first
    far = LongTermMemory(WIDTH, 4, 10, cosine=True, revise=0.999, familiar=0.01)
    far.write(np.eye(WIDTH)[0], np.eye(4)[0])
    for _ in range(50):
        far.write(np.eye(WIDTH)[1], np.eye(4)[1])
    assert far.read(np.eye(WIDTH)[0], credit=False).first() == 0  # a weight on ties, near ties


def dose(spans, doses=(1, 2, 4, 8, 16), patterns=10, seed=1):
    """Recall from noisy cues after each idle span, by the tellings a record had."""
    rng = np.random.default_rng(seed)
    store = sensory(blend=True, keep=8.0, forget=50.0)
    cleans = unit(rng.standard_normal((len(doses) * patterns, WIDTH)))
    for i, clean in enumerate(cleans):
        for _ in range(doses[i // patterns]):
            store.write(take(rng, clean), np.eye(64)[i])
    out, lived = {}, 0
    for span in spans:
        for _ in range(span - lived):
            store.moment()
        lived = span
        out[span] = [np.mean([recalled(store, rng, cleans[d * patterns + j], d * patterns + j)
                              for j in range(patterns)]) for d in range(len(doses))]
    return out, store


def test_a_record_told_more_often_is_recalled_after_a_longer_idle_span():  # G1
    curves, store = dose((0, 200, 1000))
    assert curves[0] == [1.0] * 5  # every record read before the span
    for span in (200, 1000):
        assert all(a <= b for a, b in zip(curves[span], curves[span][1:], strict=False))
    assert curves[200][0] == 0.0 and curves[200][2:] == [1.0, 1.0, 1.0]
    assert curves[1000][4] - curves[1000][0] == 1.0
    assert store.kept.sum() == 20 and store.kept[: store.count].tolist() == [
        d >= 8 for d in np.repeat((1, 2, 4, 8, 16), 10)]


def test_one_surprising_write_is_kept_as_a_record_told_eight_times_and_a_plain_one_fades():  # G2
    rng = np.random.default_rng(2)
    store = sensory(blend=True, keep=8.0, forget=50.0)
    told, struck, plain = unit(rng.standard_normal((3, WIDTH)))
    for _ in range(8):
        store.write(take(rng, told), np.eye(64)[0])
    store.write(take(rng, struck), np.eye(64)[1], surprise=7.0)  # one write, a "trauma"
    store.write(take(rng, plain), np.eye(64)[2], surprise=0.2)  # one write, little surprise
    assert store.deep[:3].tolist() == [8.0, 8.0, 1.2] and store.kept[:3].tolist() == [
        True, True, False]
    for _ in range(5000):
        store.moment()
    assert recalled(store, rng, told, 0) and recalled(store, rng, struck, 1)
    assert not recalled(store, rng, plain, 2) and store.gone[:3].tolist() == [False, False, True]
    with pytest.raises(ValueError, match="surprise"):
        store.write(told, np.eye(64)[0], surprise=-1.0)


def test_unused_records_below_the_threshold_are_forgotten_and_kept_ones_read_as_before():  # G3
    curves, store = dose((0, 1000), seed=3)
    weak = store.deep[: store.count] < 8
    assert store.gone[: store.count].tolist() == (
        weak & (store.faded[: store.count] >= 50)).tolist()
    assert curves[1000][3:] == curves[0][3:] == [1.0, 1.0]  # kept: read as before the span
    assert store.faded[: store.count][~weak].max() == 0.0  # a kept record does not fade
    for slot in np.flatnonzero(store.gone[: store.count]):  # a forgotten record is read by none
        assert store.read(store.keys[slot], credit=False).first() != slot


def test_under_a_full_budget_kept_records_survive_and_the_replaced_are_idle_and_weak():  # G6
    rng = np.random.default_rng(4)
    store = LongTermMemory(WIDTH, 64, 60, cosine=True, revise=0.999, keep=8.0)
    keys = unit(rng.standard_normal((60, WIDTH)))
    for i in range(20):  # kept: told eight times
        for _ in range(8):
            store.write(keys[i], np.eye(64)[i])
    for i in range(20, 60):  # weak: told once
        store.write(keys[i], np.eye(64)[i])
    used = np.arange(20, 30)  # weak records recall reads every moment

    def live():
        for i in used:
            store.read(keys[i])
        store.moment()

    for _ in range(4):
        live()
    assert store.count == 60 and store.kept.sum() == 20
    new = unit(rng.standard_normal((30, WIDTH)))
    taken = []
    for k in new:
        taken.append(store.write(k, np.zeros(64))[0])
        live()
    assert sorted(taken) == list(range(30, 60))  # the idle weak records, and no other
    for i in range(30):  # kept and used records hold what they held
        assert int(np.argmax(store.values[i])) == i and np.array_equal(store.keys[i], keys[i])
    full = LongTermMemory(WIDTH, 4, 3, cosine=True, revise=0.999, keep=1.0)
    for i in range(3):
        full.write(np.eye(WIDTH)[i], np.eye(4)[i], surprise=float(i))  # kept, depths 1 to 3
    assert full.write(np.eye(WIDTH)[5], np.eye(4)[3])[0] == 0  # all kept: the shallowest goes


def test_a_store_reads_a_write_s_surprise_from_its_nearest_record():
    store = LongTermMemory(WIDTH, 4, 10, cosine=True, revise=0.9, novelty=4.0)
    store.write(np.eye(WIDTH)[0], np.eye(4)[0])  # an empty store: wholly new
    assert store.deep[0] == 5.0
    half = unit(np.eye(WIDTH)[0] + np.eye(WIDTH)[1] * np.sqrt(3.0))  # cosine .5 to the first
    store.write(half, np.eye(4)[1])
    assert store.deep[1] == pytest.approx(3.0)
    store.write(half, np.eye(4)[1])  # told again: nothing new
    assert store.deep[1] == pytest.approx(4.0)
    store.write(np.eye(WIDTH)[7], np.eye(4)[2], surprise=0.0)  # the caller's surprise rules
    assert store.deep[2] == 1.0


def workload(store, seed=5):
    rng = np.random.default_rng(seed)
    keys = unit(rng.standard_normal((40, WIDTH)))
    out = []
    for i in range(300):
        k = keys[int(rng.integers(40))]
        out.append(store.write(take(rng, k, 0.1), rng.standard_normal(4)))
        if i % 3 == 0:
            read = store.read(take(rng, k, 0.3), heads=2)
            out += [read.slots.tolist(), [w.tolist() for w in read.weights]]
        store.moment()
    return out, store.arrays(), store.to_dict()


def test_every_setting_off_is_the_store_without_them():  # G5
    plain = LongTermMemory(WIDTH, 4, 30, cosine=True, revise=0.8, temperature=0.2)
    off = LongTermMemory(WIDTH, 4, 30, cosine=True, revise=0.8, temperature=0.2, blend=False,
                         familiar=0.0, keep=0.0, forget=0.0, novelty=0.0)
    a, b = workload(plain), workload(off)
    assert a[0] == b[0] and a[2] == b[2] and sorted(a[1]) == sorted(b[1])
    for name in a[1]:
        np.testing.assert_array_equal(a[1][name], b[1][name])
    assert not {"blend", "familiar", "keep", "forget", "novelty"} & set(plain.to_dict())
    assert "kept" not in a[1] and "gone" not in a[1]
    for settings in ({"blend": 1, "cosine": True}, {"familiar": -0.1}, {"keep": float("nan")},
                     {"forget": -1.0}, {"novelty": True}, {"keep": 2.0, "fade": False},
                     {"forget": 2.0, "fade": False}):
        with pytest.raises(ValueError):
            LongTermMemory(WIDTH, 4, 10, **settings)


def test_a_saved_store_keeps_what_it_kept_and_what_it_forgot():
    settings = {"cosine": True, "revise": 0.8, "blend": True, "familiar": 0.02, "keep": 4.0,
                "forget": 3.0, "novelty": 1.0}
    store = LongTermMemory(WIDTH, 4, 30, **settings)
    _, arrays, described = workload(store)
    assert {"kept", "gone"} <= set(arrays) and arrays["kept"].any() and arrays["gone"].any()
    assert {k: described[k] for k in settings} == settings
    twin = LongTermMemory(**described)
    twin.restore(arrays)
    left, right = workload(twin, seed=7), workload(store, seed=7)
    assert left[0] == right[0]
    for name in left[1]:
        np.testing.assert_array_equal(left[1][name], right[1][name])


def test_a_synaptic_memory_keeps_the_records_used_past_its_kept_depth():
    keys, value = np.eye(4), np.array([[1.0, 0.0]])
    stores = {keep: SynapticMemory(np.arange(4), np.arange(4, 6), fade=0.01, fade_keep=keep)
              for keep in (0.0, 20.0)}
    for store in stores.values():
        store.reset(1)
        for _ in range(50):  # coordinate 0 used fifty times, coordinate 1 once
            store.observe(keys[[0]], value)
            store.fade_moment(keys[[0]])
        store.observe(keys[[1]], value)
    held = {keep: store.consolidated.copy() for keep, store in stores.items()}
    for store in stores.values():
        for _ in range(40000):  # idle long past the recent trace
            store.fade_moment(None)
    kept, plain = stores[20.0], stores[0.0]
    assert kept.fade_depth[0] >= 20.0 > kept.fade_depth[1]
    assert kept.consolidated[0, 0] == held[20.0][0, 0]  # kept for good
    assert plain.consolidated[0, 0] < 0.5 * held[0.0][0, 0]  # without it, the fade takes it
    assert kept.consolidated[1, 0] < 0.5 * held[20.0][1, 0]  # under the kept depth: faded
    assert kept.to_dict()["fade_keep"] == 20.0 and "fade_keep" not in plain.to_dict()
    for settings in ({"fade_keep": 5.0}, {"fade": 0.01, "fade_keep": -1.0}):
        with pytest.raises(ValueError):
            SynapticMemory(np.arange(4), np.arange(4, 6), **settings)


# -- born in a brain: its records (``longterm_strength``) and its place records (``memory_keep``)

INPUTS, TURN = 15, 14  # agents 0-5, verbs 6-9, patients 10-13, the turn 14


def moment(*on):
    x = np.zeros((1, INPUTS))
    x[:, list(on)] = 1.0
    return x


def records_brain(**settings):
    import cadence as cd

    return cd.Brain.compose(INPUTS, 4, modules=(32,), seed=0, backend="cpu", message=70,
                            message_boundary=TURN, message_boundary_holds=True, longterm=60,
                            **settings)


def tell(brain, agent, verb, patient):
    brain.reset()
    for x in (moment(agent), moment(6 + verb), moment(TURN), moment(10 + patient)):
        brain.act(x, greedy=True)


def asked(brain, agent, verb):
    """The slot the question's read held (-1 for none)."""
    brain.reset()
    for x in (moment(agent), moment(6 + verb), moment(TURN)):
        brain.act(x, greedy=True)
    brain.stimulus(np.zeros((1, INPUTS)))
    return int(brain.longterm.read_slots[0, 0])


def checkpoint_format(path):
    import json

    with np.load(path, allow_pickle=False) as archive:
        return json.loads(str(archive["brain"]))["format"]


def test_a_brain_s_records_grow_strong_by_retelling_and_surprise_and_the_rest_fade(tmp_path):
    import cadence as cd

    brain = records_brain(longterm_strength=True)
    records = brain.longterm
    assert records.strength and not records.batched
    for _ in range(10):  # told ten times: each telling as the record holds itself deepens it
        tell(brain, 0, 0, 1)
    tell(brain, 1, 1, 2)  # told once
    brain.surprise = np.full((1, 5), 9.0)  # the brain's surprise (#337): nine times the usual
    tell(brain, 2, 2, 3)  # told once, at a moment that surprising: eight tellings more
    brain.surprise = None
    store = records.stores[0]
    assert store.count == 3 and records.counts["agreed"] == 9
    assert store.kept[:3].tolist() == [True, False, True]
    assert (store.keep, store.forget) == (records.KEEP, store.hold) == (8.0, 480.0)
    brain.reset()  # a new episode: no turn holds a record in mind
    for _ in range(520):  # idle moments: nothing is told or asked
        brain.act(np.zeros((1, INPUTS)), greedy=True)
    assert store.gone[:3].tolist() == [False, True, False]
    assert asked(brain, 0, 0) == 0 and asked(brain, 2, 2) == 2 and asked(brain, 1, 1) == -1
    assert brain.describe()["genes"]["longterm_strength"] is True
    path = brain.save(tmp_path / "strong.npz")
    assert checkpoint_format(path) == "cadence-brain/6"
    loaded = cd.Brain.load(path)
    again = loaded.longterm.stores[0]
    assert loaded.longterm.strength and again.kept[:3].tolist() == store.kept[:3].tolist()
    assert again.gone[:3].tolist() == store.gone[:3].tolist()
    np.testing.assert_array_equal(again.deep[:3], store.deep[:3])
    plain = records_brain()
    assert not plain.longterm.strength and "longterm_strength" not in plain.describe()["genes"]
    for settings in ({"longterm_strength": 1}, {"longterm_strength": True, "longterm_fade": False}):
        with pytest.raises(ValueError):
            records_brain(**settings)


def test_a_brain_s_place_records_used_past_the_kept_depth_stop_fading(tmp_path):
    import cadence as cd

    brain = cd.Brain.compose(6, 3, seed=2, arousal=True, recall=False, memory_fade=1e-3,
                             memory_keep=50.0)
    assert brain.hippocampus.fade_keep == 50.0
    assert brain.describe()["genes"]["memory_keep"] == 50.0
    path = brain.save(tmp_path / "kept.npz")
    assert checkpoint_format(path) == "cadence-brain/6"
    assert cd.Brain.load(path).hippocampus.fade_keep == 50.0
    faded = cd.Brain.compose(6, 3, seed=2, arousal=True, recall=False, memory_fade=1e-3)
    assert faded.hippocampus.fade_keep == 0.0 and "memory_keep" not in faded.describe()["genes"]
    assert checkpoint_format(faded.save(tmp_path / "faded.npz")) == "cadence-brain/3"
    with pytest.raises(ValueError, match="memory_fade"):
        cd.Brain.compose(6, 3, seed=2, recall=False, memory_keep=50.0)
