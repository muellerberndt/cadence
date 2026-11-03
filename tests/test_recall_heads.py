"""Several held heads over one recall bank (``recall_heads``): ports, addresses, lessons,
reward eligibility, checkpoints and the attention recipe."""

from __future__ import annotations

import json

import numpy as np
import pytest

import cadence as cd

STREAM = [np.eye(4)[[i]] for i in (2, 0, 3, 1)]


def composed(seed=3, heads=2, **options):
    """A small brain with the recall memory and ``heads`` heads learning their addresses."""
    options.setdefault("recall_address", 0.01)
    return cd.Brain.compose(
        4, 2, modules=(8,), seed=seed, episodic=False, recall=True, recall_heads=heads, **options
    )


def witnessed(brain, moments=STREAM):
    for x in moments:
        brain.act(x, greedy=True)
    return brain


def synapse_rows(connectome):
    return {
        (int(p), int(q), int(c), float(s))
        for p, q, c, s in zip(
            np.asarray(connectome.pre), np.asarray(connectome.post),
            np.asarray(connectome.count), np.asarray(connectome.sign), strict=True,
        )
    }


def port_synapses(brain, name):
    populations = brain.connectome.populations
    pre = np.isin(np.asarray(brain.connectome.pre), populations[name])
    post = np.isin(np.asarray(brain.connectome.post), populations["association"])
    return np.flatnonzero(pre & post)


def test_each_head_adds_one_port_and_keeps_the_brain_of_fewer_heads():
    one, three = composed(heads=1), composed(heads=3)
    populations = three.connectome.populations
    for name in ("recall", "next", "recall_1", "next_1", "recall_2", "next_2"):
        assert len(populations[name]) == len(populations["recall"])
    assert three.recall.ports(3) == (
        ("recall", "next"), ("recall_1", "next_1"), ("recall_2", "next_2")
    )
    # the further ports are appended last: the brain of one head is the first part of three's
    assert synapse_rows(one.connectome) <= synapse_rows(three.connectome)
    counts = {len(port_synapses(three, name)) for name in ("recall", "recall_1", "recall_2")}
    assert len(counts) == 1
    assert three.recall.neurons == 3 * one.recall.neurons
    assert three.recall.to_dict()["heads"] == 3 and "heads" not in one.recall.to_dict()


def test_heads_must_be_a_positive_integer_and_a_construction_setting():
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="heads"):
            composed(heads=bad)
    brain = composed(heads=2)
    with pytest.raises(ValueError, match="construction"):
        brain.retune(recall_heads=3)
    assert brain.describe()["genes"]["recall_heads"] == 2


def test_the_attention_recipe_learns_addresses_with_several_heads():
    one = cd.Brain.compose(4, 2, learned_code=4, attention=True, seed=1)
    two = cd.Brain.compose(4, 2, learned_code=4, attention=True, recall_heads=2, seed=1)
    assert (one.recall.heads, one.recall.address, one.recall.temperature) == (1, 0.0, 0.1)
    assert (two.recall.heads, two.recall.address, two.recall.temperature) == (2, 0.01, 0.25)
    chosen = cd.Brain.compose(
        4, 2, learned_code=4, attention=True, recall_heads=2, recall_temperature=0.5,
        recall_address=0.02, seed=1,
    )
    assert (chosen.recall.temperature, chosen.recall.address) == (0.5, 0.02)


def test_competing_heads_read_what_the_heads_before_them_left():
    brain = witnessed(composed(heads=2))
    recall = brain.recall
    assert recall.compete and recall.to_dict()["compete"] is True
    drive = recall.stimulate(np.zeros((1, brain.connectome.n)), STREAM[0])
    first, second = recall.attended_heads
    # at birth both addresses are the key: head 1 takes of each slot at most what head 0 left,
    # with its whole mass, and its own attention's shape where the slot can give it
    assert first[0].max() > 0.9
    assert np.all(first + second <= 1.0 + 1e-12)
    np.testing.assert_allclose(second.sum(), first.sum())
    free = second[0] < 1.0 - first[0] - 1e-12
    np.testing.assert_allclose(second[0, free] / first[0, free], second[0, free][0] / first[0, free][0])
    assert np.argmax(second[0]) != np.argmax(first[0])
    (r0, _), (r1, _) = recall._head_columns  # noqa: SLF001
    np.testing.assert_allclose(drive[:, r1], recall.amplitude * second @ recall.values[0])
    assert not np.allclose(drive[:, r0], drive[:, r1])


def test_a_head_whose_slots_the_heads_before_took_reads_nothing():
    brain = witnessed(composed(heads=2), STREAM[:1])
    recall = brain.recall
    drive = recall.stimulate(np.zeros((1, brain.connectome.n)), STREAM[0])
    first, second = recall.attended_heads
    np.testing.assert_allclose(first, [[1.0]])
    np.testing.assert_allclose(second, [[0.0]])
    _, (r1, n1) = recall._head_columns  # noqa: SLF001
    assert not drive[:, r1].any() and not drive[:, n1].any()


def test_a_head_left_nothing_to_read_is_not_taught():
    brain = witnessed(composed(heads=2, seed=9), STREAM[:1])
    recall = brain.recall
    maps = [tuple(a.copy() for a in recall.head_map(h)) for h in range(2)]
    positions = [tuple(a.copy() for a in recall.position_map(h)) for h in range(2)]
    report = brain.lesson(STREAM[1], np.array([1]))
    assert not recall.attended_heads[1].any()  # head 0 took the one slot whole
    assert report["address_rows"] == 1.0  # head 0's row alone
    for a, b in zip(maps[1], recall.head_map(1), strict=True):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(positions[1], recall.position_map(1), strict=True):
        np.testing.assert_array_equal(a, b)


def test_the_share_keeps_its_mass_however_far_apart_the_logits():
    from cadence.stream import _share

    with np.errstate(divide="ignore"):
        # one slot taken whole, two left some 725 nats below it: their attention underflows
        far = _share(np.array([[0.0, -724.8, -724.5]]), np.array([[0.0, 1.0, 1.0]]), np.ones(1))
        np.testing.assert_allclose(far, [[0.0, 1 / (1 + np.exp(0.3)), 1 / (1 + np.exp(-0.3))]])
        # the head's best slot partly taken, the rest of its mass to a tail of tiny attention
        logits = np.array([[0.0, -650.0, -700.0, -740.0]])
        left = np.array([[0.07, 0.36, 0.006, 0.88]])
        shares = _share(logits, left, np.array([0.737]))
    assert np.all(shares <= left + 1e-15)
    np.testing.assert_allclose(shares.sum(), 0.737)
    np.testing.assert_allclose(shares[0, :3], left[0, :3])
    np.testing.assert_allclose(shares[0, 3], 0.737 - left[0, :3].sum())


def test_a_competing_head_learns_from_the_shares_it_read():
    brain = witnessed(composed(heads=2, seed=10))
    recall = brain.recall
    brain.stimulus(STREAM[1])
    read = recall.attended_heads[1]
    assert recall._lesson_reads(recall.last)[1] is read  # noqa: SLF001
    # without competition a head's prior is its own attention, recomputed in its lesson
    alike = witnessed(composed(heads=2, seed=10, recall_compete=False))
    alike.stimulus(STREAM[1])
    assert alike.recall._lesson_reads(alike.recall.last) == [None, None]  # noqa: SLF001


def test_a_null_slot_keeps_its_share_of_every_competing_head():
    brain = witnessed(composed(heads=2, recall_null=0.0))
    recall = brain.recall
    recall.stimulate(np.zeros((1, brain.connectome.n)), STREAM[0])
    first, second = recall.attended_heads
    own = recall.weights(recall._head_address(recall.last, 1),  # noqa: SLF001
                         recall.position_term(1, recall.last.inputs, recall.last.cortex))
    assert first.sum() < 1.0
    np.testing.assert_allclose(second.sum(), own.sum())
    assert np.all(first + second <= 1.0 + 1e-12)


def test_competing_heads_never_read_a_moment_twice_through_lessons():
    most, overlap = {}, {}
    for compete in (True, False):
        brain = witnessed(composed(heads=2, seed=5, recall_compete=compete))
        reads, shared = [], []
        for x, label in zip(STREAM * 6, [0, 1, 1, 0] * 6, strict=True):
            brain.lesson(x, np.array([label]))
            brain.act(x, greedy=True)
            first, second = brain.recall.attended_heads
            reads.append(float((first + second).max()))
            shared.append(float(np.minimum(first, second).sum()))
        most[compete], overlap[compete] = max(reads), float(np.mean(shared))
    # the heads together read each slot at most once; without competition the slot they share
    # is read nearly twice (1.83 on this stream)
    assert most[True] <= 1.0 + 1e-12 and most[False] > 1.5
    assert overlap[True] < overlap[False] - 0.1


def test_competition_is_a_runtime_option_of_a_living_brain():
    brain = witnessed(composed(heads=2))
    brain.retune(recall_compete=False)
    assert not brain.recall.compete and brain.recall.to_dict()["compete"] is False
    brain.recall.stimulate(np.zeros((1, brain.connectome.n)), STREAM[0])
    first, second = brain.recall.attended_heads
    np.testing.assert_allclose(first, second)
    with pytest.raises(ValueError, match="compete"):
        brain.retune(recall_compete=1)


def test_a_file_saved_before_heads_competed_loads_with_heads_that_read_alike(tmp_path):
    brain = witnessed(composed(heads=2, seed=6))
    path = brain.save(tmp_path / "heads")
    assert cd.Brain.load(path).recall.compete
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    saved = json.loads(str(arrays["brain"]))

    def drop(node):  # the file as 0.91.2 wrote it: no ``compete`` beside ``heads``
        if isinstance(node, dict):
            if "heads" in node:
                node.pop("compete", None)
            for value in node.values():
                drop(value)
        elif isinstance(node, list):
            for value in node:
                drop(value)

    drop(saved)
    arrays["brain"] = np.array(json.dumps(saved))
    earlier = tmp_path / "earlier.npz"
    np.savez(earlier, **arrays)
    loaded = cd.Brain.load(earlier)
    assert not loaded.recall.compete
    loaded.recall.stimulate(np.zeros((1, loaded.connectome.n)), STREAM[0])
    first, second = loaded.recall.attended_heads
    np.testing.assert_allclose(first, second)
    assert not cd.Brain.load(loaded.save(tmp_path / "again")).recall.compete


def test_heads_read_alike_without_competition():
    brain = witnessed(composed(heads=2, recall_compete=False))
    recall = brain.recall
    drive = recall.stimulate(np.zeros((1, brain.connectome.n)), STREAM[0])
    (r0, n0), (r1, n1) = recall._head_columns  # noqa: SLF001
    np.testing.assert_allclose(drive[:, r0], drive[:, r1])
    np.testing.assert_allclose(drive[:, n0], drive[:, n1])
    first, second = recall.attended_heads
    np.testing.assert_allclose(first, second)
    head0 = drive[:, r0].copy()
    # head 1's address pointed at the oldest slot's key reads that slot; head 0 reads as before
    w_x, w_c, b = recall.head_map(1)
    recall._set_map(1, w_x, w_c, b + 20.0 * recall.keys[0, 0])  # noqa: SLF001
    drive = recall.stimulate(np.zeros((1, brain.connectome.n)), STREAM[0])
    unchanged, second = recall.attended_heads
    np.testing.assert_allclose(unchanged, first)
    np.testing.assert_allclose(drive[:, r0], head0)
    assert np.argmax(second[0]) == 0 and second[0, 0] > 0.99
    np.testing.assert_allclose(drive[:, r1], recall.amplitude * recall.values[0, 0][None, :],
                               atol=0.05 * recall.amplitude)
    assert not np.allclose(drive[:, r0], drive[:, r1])


def test_the_address_lesson_reads_each_head_back_through_its_own_synapses():
    brain = witnessed(composed(heads=2, seed=4))
    recall, graph = brain.recall, brain.learner.brain
    movement = np.random.default_rng(0).normal(size=(1, len(recall.association)))
    efficacy = np.asarray(graph.efficacy) * np.asarray(graph._gain_pre)  # noqa: SLF001
    populations = brain.connectome.populations
    local = {int(n): i for i, n in enumerate(populations["association"])}
    readbacks = []
    for h, (rname, nname) in enumerate(recall.ports(2)):
        for region, name in (("recall", rname), ("next", nname)):
            expected = np.zeros((1, len(populations[name])))
            source = {int(n): i for i, n in enumerate(populations[name])}
            for s in port_synapses(brain, name):
                pre, post = int(brain.connectome.pre[s]), int(brain.connectome.post[s])
                expected[0, source[pre]] += efficacy[s] * movement[0, local[post]]
            got = recall.readback(graph, region, movement, h)
            np.testing.assert_allclose(got, expected, atol=1e-12)
            readbacks.append(got)
    assert not np.allclose(readbacks[0], readbacks[2])  # the two heads' recall ports differ
    before = [tuple(a.copy() for a in recall.head_map(h)) for h in range(2)]
    report = brain.lesson(STREAM[2], np.array([1]))
    assert report["address_rows"] == 2.0  # one row per head
    for h in range(2):
        assert any(
            not np.array_equal(a, b) for a, b in zip(before[h], recall.head_map(h), strict=True)
        )


def test_heads_learn_different_maps_from_the_same_lesson():
    brain = witnessed(composed(heads=2, seed=5))
    for x, label in zip(STREAM * 3, [0, 1, 1, 0] * 3, strict=True):
        brain.lesson(x, np.array([label]))
        brain.act(x, greedy=True)
    maps = [brain.recall.head_map(h) for h in range(2)]
    assert any(np.abs(a).sum() > 0 for a in maps[1])
    assert not all(np.allclose(a, b) for a, b in zip(maps[0], maps[1], strict=True))


def test_a_saved_brain_of_several_heads_continues_the_same_life(tmp_path):
    brain = witnessed(composed(heads=3, seed=6))
    for x, label in zip(STREAM, [0, 1, 0, 1], strict=True):
        brain.lesson(x, np.array([label]))
        brain.act(x, greedy=True)
    path = brain.save(tmp_path / "heads")
    loaded = cd.Brain.load(path)
    assert loaded.recall.heads == 3
    for h in range(3):
        for a, b in zip(brain.recall.head_map(h), loaded.recall.head_map(h), strict=True):
            np.testing.assert_array_equal(a, b)
    for x in STREAM:
        np.testing.assert_array_equal(brain.act(x, greedy=True), loaded.act(x, greedy=True))
    # a file that lost a head's map is refused
    with np.load(path, allow_pickle=False) as data:
        arrays = {
            name: data[name] for name in data.files if name != "memories/recall/w_x_2"
        }
    broken = tmp_path / "broken.npz"
    np.savez(broken, **arrays)
    with pytest.raises(ValueError):
        cd.Brain.load(broken)


def test_reward_keeps_one_block_of_eligibility_per_head():
    brain = composed(heads=2, seed=7, recall_address_reward=0.5, arousal=True)
    witnessed(brain)
    recall = brain.recall
    before = [tuple(a.copy() for a in recall.head_map(h)) for h in range(2)]
    moved, pending = False, False
    for x in STREAM * 4:
        brain.step(x, reward=np.array([1.0]) if pending else None)
        pending = True
        moved = moved or any(
            not np.array_equal(a, b)
            for h in range(2) for a, b in zip(before[h], recall.head_map(h), strict=True)
        )
    trace = brain.basal_ganglia.trace_memory
    assert trace is not None
    width = recall.key_width + len(recall.association) + 1
    assert trace.shape[1:] == (width, 2 * recall.address_width)
    assert moved


def test_an_imagined_branch_reads_every_head_and_writes_nothing():
    brain = witnessed(composed(heads=2, seed=8))
    recall = brain.recall
    maps = [tuple(a.copy() for a in recall.head_map(h)) for h in range(2)]
    keys = recall.keys.copy()
    brain.imagine([STREAM[0], STREAM[1]])
    np.testing.assert_array_equal(recall.keys, keys)
    for h in range(2):
        for a, b in zip(maps[h], recall.head_map(h), strict=True):
            np.testing.assert_array_equal(a, b)
