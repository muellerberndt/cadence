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


def checkpoint_features(path):
    import json

    with np.load(path, allow_pickle=False) as archive:
        return json.loads(str(archive["brain"])).get("features")


def test_a_brain_s_records_grow_strong_by_retelling_and_surprise_and_the_rest_fade(tmp_path):
    import cadence as cd

    brain = records_brain(longterm_strength=True, surprise=True)
    records = brain.longterm
    assert records.strength and not records.batched
    for _ in range(4):  # a life before the facts: every word heard, no turn, so nothing written
        brain.reset()
        for word in range(TURN):
            brain.act(moment(word), greedy=True)
    assert records.stores[0].count == 0
    for _ in range(10):  # told ten times: each telling as the record holds itself deepens it
        tell(brain, 0, 0, 1)
    tell(brain, 1, 1, 2)  # told once, at a moment as surprising as the brain is used to
    assert brain.surprise.moment[0] < 0.1
    # told once, at a moment of the brain's own surprise (#337): the statement comes with an
    # input over nine times what its neuron tolerates, which counts as eight tellings and more
    brain.reset()
    for x in (moment(2), moment(6 + 2), moment(TURN)):
        brain.act(x, greedy=True)
    struck = moment(10 + 3)
    struck[0, 5] = 80.0
    brain.act(struck, greedy=True)
    assert np.expm1(brain.surprise.moment[0]) > 8.0
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
    assert checkpoint_format(path) == "cadence-brain/5"
    assert checkpoint_features(path) == ["strength", "surprise"]
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
    assert checkpoint_format(path) == "cadence-brain/5"
    assert checkpoint_features(path) == ["strength"]
    assert cd.Brain.load(path).hippocampus.fade_keep == 50.0
    faded = cd.Brain.compose(6, 3, seed=2, arousal=True, recall=False, memory_fade=1e-3)
    assert faded.hippocampus.fade_keep == 0.0 and "memory_keep" not in faded.describe()["genes"]
    plain_path = faded.save(tmp_path / "faded.npz")
    assert checkpoint_format(plain_path) == "cadence-brain/3"
    assert checkpoint_features(plain_path) is None


def test_a_checkpoint_names_its_features_and_a_reader_refuses_one_it_does_not_hold(tmp_path):
    """One format, ``cadence-brain/5``, for every brain 0.99.1 cannot continue: its record
    names the features the file needs, and a name this version does not hold is refused."""
    import json

    import cadence as cd

    brain = records_brain(longterm_strength=True)
    tell(brain, 0, 0, 1)
    path = brain.save(tmp_path / "strong.npz")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    record = json.loads(str(arrays["brain"]))
    assert record["format"] == "cadence-brain/5" and record["features"] == ["strength"]

    def written(change, name):
        changed = json.loads(json.dumps(record))
        change(changed)
        target = tmp_path / name
        np.savez(target, **{**arrays, "brain": np.array(json.dumps(changed))})
        return target

    later = written(lambda r: r["features"].append("telepathy"), "later.npz")
    with pytest.raises(ValueError, match="telepathy"):
        cd.Brain.load(later)
    unnamed = written(lambda r: r.pop("features"), "unnamed.npz")
    with pytest.raises(ValueError, match="cadence-brain/5"):
        cd.Brain.load(unnamed)
    older = written(lambda r: r.update(format="cadence-brain/4") or r.pop("features"), "older.npz")
    with pytest.raises(ValueError, match="cadence-brain/5"):
        cd.Brain.load(older)
    assert cd.Brain.load(path).longterm.strength
    with pytest.raises(ValueError, match="memory_fade"):
        cd.Brain.compose(6, 3, seed=2, recall=False, memory_keep=50.0)


# -- with records read by what tells them apart (``longterm_distinct``)


def distinct_strong(**settings):
    import cadence as cd

    return cd.Brain.compose(INPUTS, 4, modules=(32,), seed=0, backend="cpu", message=70,
                            longterm=16, longterm_turn=TURN, longterm_distinct=0.5,
                            longterm_heads=2, longterm_match=0.5, longterm_strength=True,
                            **settings)


def say(brain, *words):
    for on in words:
        brain.act(moment(on) if on is not None else np.zeros((1, INPUTS)), greedy=True)


def answered(brain, agent):
    """The slot the first head holds for a question that names ``agent`` (-1 for none)."""
    brain.reset()
    say(brain, agent, 8, TURN, None)  # the question, then the moment its answer is read
    return int(brain.longterm.read_slots[0, 0])


def test_distinct_records_grow_strong_by_use_and_an_unused_one_is_forgotten(tmp_path):
    import cadence as cd

    brain = distinct_strong()
    records = brain.longterm
    brain.reset()
    say(brain, 0, 6, TURN, 10)  # agent 0 with verb 0: patient 10
    say(brain, 1, 6, TURN, 11)  # agent 1: told once, never asked about
    store = records.stores[0]
    assert store.count == 2 and not store.kept[:2].any()
    for _ in range(30):  # asked about again and again: each read that delivers it deepens it
        if store.kept[0]:
            break
        assert answered(brain, 0) == 0
    assert store.kept[:2].tolist() == [True, False] and store.deep[0] > store.deep[1]
    brain.reset()
    for _ in range(4000):  # idle moments, until the record told once is forgotten
        if store.gone[1]:
            break
        say(brain, None)
    assert store.gone[:2].tolist() == [False, True]
    assert answered(brain, 1) == -1 and answered(brain, 0) == 0  # no read finds the forgotten one
    path = brain.save(tmp_path / "distinct-strong.npz")
    again = cd.Brain.load(path)
    assert answered(again, 1) == -1 and answered(again, 0) == 0
    brain.reset()
    say(brain, 2, 7, TURN, 12)  # a new fact is a record of its own, and is read
    assert answered(brain, 2) == 2 and answered(brain, 1) == -1



def test_a_refused_moment_deepens_no_record_and_a_moment_deepens_what_it_used_once():
    """With strength, the records a read delivers deepen once the moment is accepted: a refused
    act leaves every depth as it was, and step (a lesson's read, then the act's) deepens a
    record as act does."""
    def asked_twice(call):
        brain = distinct_strong()
        brain.reset()
        say(brain, 0, 6, TURN, 10)  # two facts, as the test above tells them
        say(brain, 1, 6, TURN, 11)
        store = brain.longterm.stores[0]
        brain.reset()
        say(brain, 0, 8, TURN)  # a question about agent 0: read at the moment after the turn
        before = store.deep[: store.count].copy()
        if call == "refused":
            brain.retune(learning_free_steps=0)  # the next settle is refused
            with pytest.raises(RuntimeError, match="settle"):
                brain.act(np.zeros((1, INPUTS)), greedy=True)
            return before, store.deep[: store.count].copy()
        silent = np.zeros((1, INPUTS))  # the moment after the turn: the records are read
        if call == "act":
            brain.act(silent, greedy=True)
        else:
            brain.step(silent, teacher=np.array([0]))
        return before, store.deep[: store.count].copy()

    before, after = asked_twice("refused")
    np.testing.assert_array_equal(before, after)
    gained = {call: asked_twice(call) for call in ("act", "step")}
    act_gain = gained["act"][1] - gained["act"][0]
    step_gain = gained["step"][1] - gained["step"][0]
    assert act_gain.max() > 0.5
    np.testing.assert_allclose(step_gain, act_gain, atol=1e-9)


def test_a_brain_with_surprise_or_a_map_of_values_names_them_in_its_file(tmp_path):
    """A reader of 0.99.1 cannot build a brain's surprise or a map whose sheet carries values;
    their files are cadence-brain/5 and name them."""
    import json

    import cadence as cd
    from cadence.space.map import Map

    def record(brain, name):
        with np.load(brain.save(tmp_path / name), allow_pickle=False) as archive:
            return json.loads(str(archive["brain"]))

    surprised = cd.Brain.compose(INPUTS, 4, seed=0, backend="cpu", surprise=True)
    surprised.act(moment(0))
    saved = record(surprised, "surprised.npz")
    assert saved["format"] == "cadence-brain/5" and saved["features"] == ["surprise"]
    assert cd.Brain.load(tmp_path / "surprised.npz").surprise is not None
    painted = cd.Brain.compose(INPUTS, 4, seed=0, backend="cpu")
    painted.space = Map((-1.0, 1.0, -1.0, 1.0), (8, 8), units=4, values={"colour": 3})
    assert record(painted, "painted.npz")["features"] == ["map"]
    plain = cd.Brain.compose(INPUTS, 4, seed=0, backend="cpu")
    plain.space = Map((-1.0, 1.0, -1.0, 1.0), (8, 8), units=4)
    assert record(plain, "plain.npz")["format"] == "cadence-brain/1"
