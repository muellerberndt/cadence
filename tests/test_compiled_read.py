"""The recall read's episode terms and a greedy choice in compiled passes: the same bits as the
NumPy forms they stand in for."""

from __future__ import annotations

import numpy as np
import pytest

import cadence as cd
from cadence import stream

pytest.importorskip("numba")


def composed() -> cd.Brain:
    return cd.Brain.compose(6, 3, modules=(16,), seed=2, recall=True, recall_recency=0.0,
                            recall_anchor=True, recall_relative=True, recall_slope=0.1,
                            recall_capacity=24, episodic=False)


def test_the_episode_terms_are_the_numpy_terms(monkeypatch: pytest.MonkeyPatch) -> None:
    brain = composed()
    recall = brain.recall
    rng = np.random.default_rng(0)
    for moment in range(40):
        x = np.eye(6)[rng.integers(0, 6, 3)]
        brain.act(x, greedy=True)
        if moment == 12:
            recall.reset(3, rows=np.array([1]))  # a stream that begins again holds fewer slots
        event = rng.random((3, 6))
        cortex = rng.random((3, 16))
        compiled = recall.position_term(0, event, cortex)
        rows = np.array([2, 0])
        some = recall.position_term(0, event[rows], cortex[rows], rows)
        with monkeypatch.context() as patched:
            patched.setattr(stream, "_compiled_terms", lambda: False)
            plain = recall.position_term(0, event, cortex)
            plain_some = recall.position_term(0, event[rows], cortex[rows], rows)
        assert compiled.tobytes() == plain.tobytes()
        assert some.tobytes() == plain_some.tobytes()
    # places past the table's end, as a long stream's are
    recall.told = recall.told + 500
    event, cortex = rng.random((3, 6)), rng.random((3, 16))
    compiled = recall.position_term(0, event, cortex)
    with monkeypatch.context() as patched:
        patched.setattr(stream, "_compiled_terms", lambda: False)
        assert compiled.tobytes() == recall.position_term(0, event, cortex).tobytes()


def softmax_choice(s: np.ndarray, temperature: float) -> np.ndarray:
    z = s / temperature
    p = np.exp(z - z.max(axis=-1, keepdims=True))
    return np.argmax(p / p.sum(axis=-1, keepdims=True), axis=-1)


def test_a_clear_choice_is_the_softmax_choice_and_a_near_tie_declines() -> None:
    from cadence.fused import clear_choice

    rng = np.random.default_rng(1)
    s = rng.random((64, 7))
    np.testing.assert_array_equal(clear_choice(s, 0.2), softmax_choice(s, 0.2))
    near = s.copy()
    near[5, 3] = near[5].max() + 1e-14
    assert clear_choice(near, 0.2) is None
    level = s.copy()
    level[9, :] = 0.5
    assert clear_choice(level, 0.2) is None
    wild = s.copy()
    wild[2, 2] = np.inf
    assert clear_choice(wild, 0.2) is None
    wild[2, 2] = np.nan
    assert clear_choice(wild, 0.2) is None
    assert clear_choice(np.array([[0.3], [0.1]]), 0.2).tolist() == [0, 0]


def test_a_greedy_act_reads_as_it_did_at_a_tie() -> None:
    brain = cd.Brain.compose(4, 3, modules=(8,), seed=0, episodic=False)
    agent = brain.basal_ganglia
    x = np.eye(4)[[0, 1]]
    drive = brain.stimulus(x)
    state = brain.brain.settle_batch(drive, steps=4)
    out = brain.learner.output_index
    # the second row's first two motor neurons within rounding of each other
    state.activation[1, out[1]] = np.nextafter(state.activation[1, out[0]], np.inf)
    state.activation[1, out[2]] = -1.0
    expected = np.argmax(agent.probabilities(state), axis=-1)
    agent._free, agent._free_brain, agent._drive = state, brain.brain, drive.copy()
    np.testing.assert_array_equal(agent.act(drive, greedy=True), expected)


def test_the_readback_is_the_one_a_sum_over_the_synapses_makes() -> None:
    brain = composed()
    recall = brain.recall
    graph = brain.brain
    movement = np.random.default_rng(3).normal(size=(3, 16))
    for region in ("recall", "next"):
        read = recall.readback(graph, region, movement)
        synapses, rows, cols = recall._head_synapses[0][0 if region == "recall" else 1]
        width = len(recall._heads[0][0 if region == "recall" else 1])
        w = np.zeros((width, 16))
        np.add.at(w, (rows, cols), graph._gain_pre[synapses] * graph.efficacy[synapses])
        assert read.tobytes() == (movement @ w.T).tobytes()
