"""Contracts of the index memory."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cadence import Brain, PatternSeparator

HERE = Path(__file__).resolve().parent


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ir = _load("index_read")
tb = ir.tb


def _walk(brain, seed: int, rows: int = 4, cut: int = 1):
    trials = tb.draw(np.random.default_rng(seed), rows, tb.holdout(seed))
    seq = tb.events(trials)
    brain.reset()
    for x in seq[:-cut]:
        brain.act(x, greedy=True)
    return trials, seq


def test_without_read_regions_the_brain_is_the_composed_one() -> None:
    built = ir.IndexBrain.build(3)
    composed = Brain.compose(tb.INPUTS, tb.VALUES, modules=(tb.WIDTH,), lateral=tb.LATERAL, seed=3)
    np.testing.assert_array_equal(built.brain.efficacy, composed.brain.efficacy)
    assert built.index is None


def test_an_empty_memory_acts_as_the_composed_brain() -> None:
    indexed, plain = ir.IndexBrain.build(1, read={}), ir.IndexBrain.build(1)
    n = plain.connectome.n
    pre, post = np.asarray(indexed.connectome.pre), np.asarray(indexed.connectome.post)
    shared = (pre < n) & (post < n)  # the composed brain's synapses, in the same order
    np.testing.assert_array_equal(pre[shared], plain.connectome.pre)
    np.testing.assert_array_equal(indexed.brain.efficacy[shared], plain.brain.efficacy)
    trials = tb.draw(np.random.default_rng(1), 4, tb.holdout(1))
    first = tb.events(trials)[0]
    for brain in (indexed, plain):
        brain.reset()
    np.testing.assert_array_equal(indexed.act(first, greedy=True), plain.act(first, greedy=True))
    a = np.atleast_2d(indexed.basal_ganglia.state.activation)
    b = np.atleast_2d(plain.basal_ganglia.state.activation)
    np.testing.assert_allclose(a[:, :n], b, atol=1e-12)
    assert not a[:, n:].any()  # the recall and next neurons are silent


def test_the_address_starts_as_the_event_heard_and_reads_by_competition() -> None:
    brain = ir.IndexBrain.build(2, read={"heard": 2.0})
    _, seq = _walk(brain, 2, cut=2)  # two facts heard
    index = brain.index
    memory = index.slots
    assert memory.slots == 2
    np.testing.assert_allclose(memory.keys[:, 0], 2.0 * seq[0])
    np.testing.assert_allclose(memory.values[:, 1], seq[1])
    np.testing.assert_allclose(memory.nexts[:, 0], seq[1])
    assert not memory.nexts[:, 1].any()
    q = index.address(seq[3])  # the query: the asked name heard
    np.testing.assert_allclose(q, 2.0 * seq[3])
    logits = np.einsum("bk,bmk->bm", q, memory.keys) / index.temperature
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(index.weights(q), p)
    drive = brain.stimulus(seq[3])
    np.testing.assert_allclose(drive[:, brain.index_recall],
                               3.0 * np.einsum("bm,bmc->bc", p, memory.values))
    np.testing.assert_allclose(drive[:, brain.index_next],
                               3.0 * np.einsum("bm,bmc->bc", p, memory.nexts))


def test_recency_and_capacity() -> None:
    memory = ir.SlotMemory(3, 2, capacity=2)
    for t in range(4):
        memory.observe(np.full((1, 3), t, float), np.full((1, 2), 10 + t, float))
    assert memory.slots == 2
    np.testing.assert_array_equal(memory.keys[0, :, 0], [2, 3])
    np.testing.assert_array_equal(memory.nexts[0, :, 0], [13, 0])
    np.testing.assert_array_equal(memory.ages(), [1, 0])
    index = ir.IndexMemory(3, 4, 2, dict(ir.GENES, recency=1.0, temperature=1.0))
    index.slots.observe(np.ones((1, 3)), np.ones((1, 2)))
    index.slots.observe(np.ones((1, 3)), np.ones((1, 2)))
    p = index.weights(np.ones((1, 3)))
    np.testing.assert_allclose(p[0], np.exp([-1.0, 0.0]) / np.exp([-1.0, 0.0]).sum())


def test_the_readback_is_the_transpose_of_the_sources_synapses() -> None:
    brain = ir.IndexBrain.build(4, read={})
    graph = brain.learner.brain
    connectome = graph.connectome
    pre, post = np.asarray(connectome.pre), np.asarray(connectome.post)
    movement = np.random.default_rng(4).normal(size=(3, len(brain.association_index)))
    got = brain.readback(brain.index_recall, movement)
    weight = graph._gain_pre * graph.efficacy
    for k, neuron in enumerate(brain.index_recall[:4]):
        out = pre == neuron
        assert out.any()
        columns = np.searchsorted(brain.association_index, post[out])
        np.testing.assert_allclose(got[:, k], movement[:, columns] @ weight[out])


def test_the_address_lesson_moves_toward_the_slot_the_movement_names() -> None:
    brain = ir.IndexBrain.build(5, read={"address": 0.5})
    _, seq = _walk(brain, 5, cut=1)  # facts and the neutral event heard
    index = brain.index
    q0 = index.address(seq[-1]).copy()
    rows = np.arange(len(q0))
    d_value = index.slots.values[:, 1].copy()  # the movement points at the second fact
    report = index.learn(rows, d_value, None, rate=0.5, temperature=0.05)
    assert report["address_rows"] == len(rows)
    target = index.slots.keys[:, 1]
    q1 = index.address(seq[-1])
    assert np.all(np.linalg.norm(q1 - target, axis=1) < np.linalg.norm(q0 - target, axis=1))


def test_the_address_lesson_reads_the_lessons_phases() -> None:
    brain = ir.IndexBrain.build(6, read={"address": 0.2})
    _, seq = _walk(brain, 6, cut=1)
    brain.stimulus(seq[-1])  # the moment's address, as a lesson computes it
    n = brain.connectome.n
    free = np.zeros((4, n))
    plus, minus = free.copy(), free.copy()
    plus[:, brain.association_index] = 0.1
    learned = SimpleNamespace(free=SimpleNamespace(activation=free),
                              nudged=SimpleNamespace(activation=plus),
                              opposite=SimpleNamespace(activation=minus))
    before = brain.index.w_x.copy()
    report = ir.address_lesson(brain, learned, np.arange(4))
    assert report["address_rows"] == 4
    assert not np.array_equal(brain.index.w_x, before)


def test_lessons_teach_the_address_at_the_taught_rows() -> None:
    brain = ir.IndexBrain.build(7, read={"address": 0.1}, learning_surprise=True)
    held = tb.holdout(7)
    rng = np.random.default_rng(7)
    work = tb.new_work()
    for _ in range(3):
        trials = tb.draw(rng, tb.BATCH, held)
        brain.reset()
        seq = tb.events(trials)
        for x in seq[:-1]:
            tb.act(brain, x, work)
        ir.lesson(brain, seq[-1], tb.answers(trials), work)
        tb.act(brain, seq[-1], work)
    assert work["lesson_refusals"] == 0
    assert 0 < work.get("address_rows", 0) <= work["rows_taught"]


def test_forks_carry_their_own_memory() -> None:
    brain = ir.IndexBrain.build(8, read={})
    _, seq = _walk(brain, 8, cut=2)
    twin = tb.fork(brain)
    assert twin.index is not brain.index
    before = brain.index.slots.keys.copy()
    twin.act(seq[2], greedy=True)
    np.testing.assert_array_equal(brain.index.slots.keys, before)
    assert twin.index.slots.slots == brain.index.slots.slots + 1


def test_the_sparse_code_keeps_its_winners_at_unit_length_whatever_the_width() -> None:
    narrow = ir.SparseCode(64, 128, 8, 6, seed=3)
    wide = ir.SparseCode(12_824, 128, 8, 6, seed=3)
    assert narrow.index.shape == (128, 8)  # eight reads per unit read every channel twice
    assert wide.index.shape == (128, 401)  # the fan-in grows until every channel is read twice
    for code in (narrow, wide):
        channels = code.separator.channels
        assert channels == 2 * code.separator.inputs and code.index.max() < channels
        assert np.bincount(code.index.ravel(), minlength=channels).min() >= 2
    x = (np.random.default_rng(3).random((5, 12_824)) < 0.05).astype(float)
    code = wide(x)
    assert np.all((code > 0).sum(axis=1) <= 6)
    np.testing.assert_allclose(np.linalg.norm(code, axis=1), 1.0)
    np.testing.assert_array_equal(code, ir.SparseCode(12_824, 128, 8, 6, seed=3)(x))
    # a negative input reaches the code through its OFF channel
    assert wide(-np.eye(12_824)[:64]).any(axis=1).all()


def test_the_sparse_code_is_the_librarys_on_a_stream_of_its_own() -> None:
    """The code's seed is drawn from the founder's on a stream of its own, as the library
    draws the brain's code: at the same inputs, units and fan-in the tables are the library
    brain's, and they are not the tables of the founder's own stream."""
    code = ir.SparseCode(12_824, 1024, 32, 24, seed=3)
    own = int(np.random.SeedSequence([3, 7103]).generate_state(1)[0])
    assert code.separator.seed == own != 3
    brain = Brain.compose(12_824, 2, modules=(8,), code=1024, seed=3, episodic=False)
    assert brain.sparse_code.seed == own and brain.sparse_code.fan_in == code.index.shape[1]
    np.testing.assert_array_equal(code.index, brain.sparse_code.index)
    np.testing.assert_array_equal(code.separator.weight, brain.sparse_code.weight)
    raw = PatternSeparator(12_824, 1024, 24, seed=3, fan_in=51, cover=2)  # the founder's stream
    assert not np.array_equal(code.index, raw.index)


def test_a_coded_brain_keys_and_values_its_moments_in_the_code() -> None:
    brain = ir.IndexBrain.build(9, read={"code": 64, "code_fan_in": 4, "code_winners": 5})
    _, seq = _walk(brain, 9, cut=2)
    memory = brain.index.slots
    assert memory.keys.shape[2] == memory.values.shape[2] == 64
    np.testing.assert_allclose(memory.values[:, 0], brain.code(seq[0]))
    np.testing.assert_allclose(memory.keys[:, 0], brain.code(seq[0]))  # W = 0: the code heard
    assert len(brain.index_recall) == 64


def test_the_answers_address_is_named_among_the_moments_attended_to() -> None:
    """A moment whose own value is the answer does not displace the moment the address found:
    with the attention as the prior the step toward it is a fraction of the step without."""

    def toward_second(prior: float) -> float:
        index = ir.IndexMemory(3, 2, 2, dict(ir.GENES, temperature=0.25, address_prior=prior))
        index.slots.observe(np.array([[1.0, 0, 0]]), np.array([[1.0, 0]]))  # the moment found
        index.slots.observe(np.array([[0, 1.0, 0]]), np.array([[0, 1.0]]))  # its value: the answer
        before = index.address(np.array([[1.0, 0, 0]])).copy()
        d_value = np.array([[0.0, 1.0]])  # the answer's movement matches the second value
        index.learn(np.array([0]), d_value, None, rate=1.0, temperature=0.5)
        after = index.address(np.array([[1.0, 0, 0]]))
        second = index.slots.keys[0, 1]
        return float((after - before)[0] @ second)

    assert 0 < toward_second(1.0) < 0.2 * toward_second(0.0)


def test_a_movement_that_names_a_short_key_raises_its_slot_under_the_toward_step() -> None:
    """Two slots with the short keys [.25, 0] and [0, .25] and the heard key [1, 0]: a
    movement that names the first slot gives a target near its short key. The founder step
    moves the address toward it and the first slot's weight falls; the toward step moves the
    address by the target less the attended keys' mixture and the weight rises."""

    def moved(form: str) -> float:
        genes = dict(ir.GENES, recency=0.0, next=False, address_step=form)
        index = ir.IndexMemory(2, 1, 2, genes)
        for key in 0.25 * np.eye(2):
            index.slots.observe(key[None], key[None])
        x = np.array([[1.0, 0.0]])
        before = index.weights(index.address(x))[0, 0]
        report = index.learn(np.array([0]), index.slots.values[:, 0].copy(), None, rate=0.01,
                             temperature=0.1)
        assert report["address_rows"] == 1.0
        return float(index.weights(index.address(x))[0, 0] - before)

    assert moved("lms") < 0.0 < moved("toward")
    assert ir.GENES["address_step"] == "lms"


def test_the_recency_default_is_a_power_of_age_and_nothing_without_a_bound() -> None:
    """With ``recency=None`` a slot's age costs ``r log(1 + age)``, ``r = heard^2 / T /
    log(1 + capacity)``: a fully matched target as old as the capacity allows keeps more
    weight than a fresh unrelated slot, two equal matches at ages 1 and 3 keep weights in
    the ratio ``2^r``, and an unbounded memory (capacity 0) charges nothing."""
    capacity = 8
    index = ir.IndexMemory(capacity, 1, capacity,
                           dict(ir.GENES, recency=None, capacity=capacity, heard=1.5))
    events = np.eye(capacity)
    for event in events:
        index.slots.observe(1.5 * event[None], event[None])
    exponent = 1.5**2 / 0.25 / np.log(1 + capacity)
    np.testing.assert_allclose(index.charges(), exponent * np.log1p(index.slots.ages()))
    p = index.weights(index.address(events[[0]]))[0]
    assert p[0] > p[-1]
    repeated = ir.IndexMemory(4, 1, 4, dict(ir.GENES, recency=None, capacity=capacity))
    for event in np.eye(4)[[0, 1, 0, 2]]:
        repeated.slots.observe(event[None], event[None])
    p = repeated.weights(repeated.address(np.eye(4)[[0]]))[0]
    np.testing.assert_allclose(p[2] / p[0], 2.0 ** (1 / 0.25 / np.log(1 + capacity)))
    unbounded = ir.IndexMemory(4, 1, 4, dict(ir.GENES, recency=None, capacity=0))
    for event in np.eye(4)[[0, 1, 0, 2]]:
        unbounded.slots.observe(event[None], event[None])
    assert not unbounded.charges().any()
    p = unbounded.weights(unbounded.address(np.eye(4)[[0]]))[0]
    np.testing.assert_allclose(p[2], p[0])


def test_aroused_rows_mark_the_streams_a_moment_is_written_for() -> None:
    """With ``write="aroused"`` a greedy act writes the streams a caller marks in
    ``aroused_rows`` and nothing unmarked; a sampled act writes every stream. A moment written
    for some streams is a column every stream has, held by the written ones (zeros elsewhere),
    and each stream's latest held slot learns that stream's next written moment."""
    brain = ir.IndexBrain.build(10, read={"write": "aroused"})
    _, seq = _walk(brain, 10, rows=3, cut=4)  # reset and nothing acted on
    memory = brain.index.slots
    brain.act(seq[0], greedy=True)  # calm and unmarked: nothing written
    assert memory.slots == 0
    brain.aroused_rows = np.array([True, False, True])
    brain.act(seq[1], greedy=True)
    brain.aroused_rows = None
    np.testing.assert_array_equal(memory.held, [[True], [False], [True]])
    assert not memory.keys[1].any() and not memory.values[1].any()
    brain.act(seq[2])  # sampled: every stream
    np.testing.assert_array_equal(memory.held, [[True, True], [False, True], [True, True]])
    np.testing.assert_array_equal(memory.nexts[[0, 2], 0], seq[2][[0, 2]])
    brain.aroused_rows = np.array([False, True, False])
    brain.act(seq[3], greedy=True)
    brain.aroused_rows = None
    np.testing.assert_array_equal(memory.nexts[1, 1], seq[3][1])  # stream 1's latest held slot
    assert not memory.nexts[[0, 2], 1].any()  # the others wrote nothing since
    p = brain.index.weights(np.ones((3, tb.INPUTS)))
    assert p[1, 0] == 0.0 and p[0, 2] == 0.0
    np.testing.assert_allclose(p.sum(axis=1), 1.0)


def test_a_null_slot_reads_nothing_and_takes_its_share() -> None:
    """Three slots and a null slot of logit 0, a query that matches nothing, no recency: the
    four logits are equal, so each slot keeps a quarter and the null slot the rest."""
    index = ir.IndexMemory(3, 1, 3, dict(ir.GENES, null=0.0))
    for k in range(3):
        index.slots.observe(np.eye(3)[[k]], np.eye(3)[[k]])
    np.testing.assert_allclose(index.weights(np.zeros((1, 3))), np.full((1, 3), 0.25))
    found = index.read(index.address(np.zeros((1, 3))))
    assert found is not None
    np.testing.assert_allclose(found[0], np.full((1, 3), 0.25))
    np.testing.assert_allclose(index.attended.sum(), 0.75)
    assert ir.GENES["null"] is None and ir.GENES["write"] == "always"
    for null, temperature in ((float("inf"), 0.25), (1e300, 1e-10)):  # a logit not finite
        with pytest.raises(ValueError, match="null"):
            ir.IndexMemory(3, 1, 3, dict(ir.GENES, null=null, temperature=temperature))
