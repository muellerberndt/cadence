"""The read's drive as one more carried route (``recall_drive_gain``): one learned gain over every
read port, starting at the read's amplitude and reaching no higher, taught by the settle's contrast as
``carried_gain`` teaches the trace's."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import cadence as cd
from cadence.stream import Hearing


def brain_of(**options) -> cd.Brain:
    return cd.Brain.compose(6, 4, modules=(16,), seed=5, **{"recall": False, **{"recall": True, "recall_founder": True, **options}})


def gained(**options) -> cd.Brain:
    return brain_of(carried_gain=0.01, **options)


def routed(**options) -> cd.Brain:
    return brain_of(**{"carried_gain": 0.01, "recall_drive_gain": True, **options})


def teach(brain: cd.Brain, seed: int = 4, lessons: int = 12) -> None:
    rng = np.random.default_rng(seed)
    for _ in range(lessons):
        brain.step(rng.random((2, 6)), teacher=rng.integers(0, 4, size=2))


def live(brain: cd.Brain, seed: int = 7, moments: int = 12) -> None:
    rng = np.random.default_rng(seed)
    brain.reset()
    for t in range(moments):
        brain.act(rng.random((2, 6)))
        brain.learn(rng.random(2), np.full(2, t % 5 == 4), rng.random((2, 6)))


def port_columns(brain: cd.Brain) -> np.ndarray:
    return np.asarray(brain.recall.port_neurons)


def test_without_it_nothing_of_it():
    plain = brain_of()
    assert not plain.recall.drive_gain and plain.recall.hearing == 1.0
    assert "read" not in plain.carried_gains
    described = plain.describe()
    assert "recall_drive_gain" not in described["genes"]
    assert "hearing" not in described["memories"]["recall"]
    assert not len(plain.recall.trace)
    x = np.random.default_rng(1).random((2, 6))
    plain.act(x, greedy=True)
    assert not len(plain.recall.trace)  # kept only for the route's lesson


def test_a_newborn_acts_and_settles_as_one_without_it():
    plain, brain = brain_of(), routed()
    rng = np.random.default_rng(2)
    for _ in range(4):
        x = rng.random((2, 6))
        np.testing.assert_array_equal(plain.stimulus(x), brain.stimulus(x))
        np.testing.assert_array_equal(plain.act(x, greedy=True), brain.act(x, greedy=True))
    assert brain.carried_gains["read"] == 1.0
    np.testing.assert_array_equal(plain.brain.efficacy, brain.brain.efficacy)


def test_it_is_one_more_route_of_the_carried_gains():
    brain = routed()
    assert brain.carried_gains == {"trace": 1.0, "read": 1.0}
    assert brain.parameters() == gained().parameters() + 1
    genes = brain.describe()["genes"]
    assert genes["recall_drive_gain"] is True and genes["carried_gain"] == 0.01
    assert brain.describe()["memories"]["recall"]["hearing"] == 1.0
    assert brain.hearing.rate == 0.01


def test_every_read_port_is_on_the_one_route():
    options = {"recall_heads": 2, "recall_when": True, "recall_action": True}
    brain = routed(**options)
    assert brain.carried_gains == {"trace": 1.0, "read": 1.0}
    assert brain.parameters() == gained(**options).parameters() + 1
    populations = brain.connectome.populations
    names = [
        "recall", "recall_action", "next", "next_action", "recall_1", "recall_1_action", "next_1",
        "next_1_action", "recall_when", "recall_1_when",
    ]
    assert set(port_columns(brain)) == {n for name in names for n in populations[name]}
    assert len(port_columns(brain)) == brain.recall.neurons
    route = brain.hearing.routes[-1]
    assert route["name"] == "read" and route["width"] == brain.recall.neurons


def test_it_needs_the_carried_gain_and_the_memory():
    with pytest.raises(ValueError, match="carried_gain"):
        routed(carried_gain=0.0)
    with pytest.raises(ValueError, match="carried_gain"):
        brain_of(recall_drive_gain=True)
    with pytest.raises(ValueError, match="recall=True"):
        brain_of(recall=False, recall_drive_gain=True)
    with pytest.raises(ValueError, match="drive_gain must be boolean"):
        brain_of(recall_drive_gain=1)


def test_the_gain_scales_every_port_and_nothing_else():
    brain = routed(recall_heads=2, recall_when=True)
    rng = np.random.default_rng(2)
    for _ in range(3):
        brain.act(rng.random((2, 6)), greedy=True)
    recall = brain.recall
    ports = port_columns(brain)
    others = np.setdiff1d(np.arange(brain.connectome.n), ports)
    x = rng.random((2, 6))
    before = brain.stimulus(x)
    assert before[:, ports].any()
    np.testing.assert_array_equal(before[:, ports], recall.amplitude * recall.trace)
    recall.hearing = 0.5
    after = brain.stimulus(x)
    np.testing.assert_allclose(after[:, ports], 0.5 * before[:, ports])
    np.testing.assert_array_equal(after[:, others], before[:, others])
    assert brain.carried_gains["read"] == 0.5


def test_a_silent_read_keeps_what_the_ports_hold_so_the_gain_can_come_back():
    brain = routed()
    rng = np.random.default_rng(3)
    for _ in range(3):
        brain.act(rng.random((2, 6)), greedy=True)
    recall = brain.recall
    x = rng.random((2, 6))
    loud = brain.stimulus(x)[:, port_columns(brain)]
    recall.hearing = 0.0
    silent = brain.stimulus(x)[:, port_columns(brain)]
    assert not silent.any()
    np.testing.assert_allclose(recall.amplitude * recall.trace, loud)
    movement = rng.normal(size=(2, len(brain.association_index)))
    c = brain.hearing._contrast(brain.learner.brain, movement, None, 2)
    # the read's weights start at the composition's scale: the contrast is alive at zero
    assert np.abs(c[:, list(brain.carried_gains).index("read")]).max() > 0


def test_the_amplitude_retunes_under_the_gain():
    brain = routed()
    rng = np.random.default_rng(4)
    for _ in range(3):
        brain.act(rng.random((2, 6)), greedy=True)
    x = rng.random((2, 6))
    before = brain.stimulus(x)[:, port_columns(brain)]
    brain.recall.hearing = 0.5
    brain.retune(recall_amplitude=1.0)
    after = brain.stimulus(x)[:, port_columns(brain)]
    np.testing.assert_allclose(after, before / 3.0 * 0.5)
    # at amplitude zero the contrast is zero and the gain holds
    brain.retune(recall_amplitude=0.0)
    teach(brain, lessons=3)
    assert brain.carried_gains["read"] == 0.5


def test_the_contrast_is_the_energys_derivative_on_the_route():
    brain = routed()
    hearing = brain.hearing
    rng = np.random.default_rng(3)
    for _ in range(3):
        brain.act(rng.random((2, 6)), greedy=True)
    brain.stimulus(rng.random((2, 6)))
    recall = brain.recall
    association = np.asarray(brain.association_index)
    movement = rng.normal(size=(2, len(association)))
    k = list(brain.carried_gains).index("read")
    got = hearing._contrast(brain.learner.brain, movement, None, 2)[:, k]
    # sum_j f'(v_j) a t_j sum_i W_ij m_i, summed by hand over the route's synapses
    graph, route = brain.learner.brain, hearing.routes[k]
    weights = np.asarray(graph._gain_pre) * np.asarray(graph.efficacy)
    slope = brain.brain.neuron_model.slope_at(recall.amplitude * recall.trace)
    expected = np.zeros(2)
    for synapse, j, i in zip(route["synapses"], route["rows"], route["cols"], strict=True):
        expected += (
            slope[:, j] * recall.amplitude * recall.trace[:, j] * weights[synapse] * movement[:, i]
        )
    np.testing.assert_allclose(got, expected)
    assert np.abs(expected).max() > 0
    # and the same through the readback the address lesson reads
    readback = np.concatenate(
        [recall.readback(graph, "recall", movement), recall.readback(graph, "next", movement)],
        axis=1,
    )
    np.testing.assert_allclose(
        got, np.einsum("bj,bj->b", slope * recall.amplitude * recall.trace, readback)
    )


def test_the_route_reads_the_read_each_act_settled_on():
    brain = routed(arousal=True)
    agent, recall = brain.basal_ganglia, brain.recall
    held = []
    rows = brain.hearing.contrast_rows

    def spy(graph, plus, minus, beta):
        held.append((agent._drive[:, port_columns(brain)].copy(), recall.trace.copy()))
        return rows(graph, plus, minus, beta)

    brain.hearing.contrast_rows = spy
    rng = np.random.default_rng(6)
    brain.live(rng.random((1, 6)))
    for t in range(9):
        brain.live(rng.random((1, 6)), reward=[1.0 if t % 2 else -1.0], done=[t % 4 == 3])
    assert len(held) >= 8
    assert any(drive.any() for drive, _ in held)
    for drive, trace in held:  # the same read, at the gain of the moment it was taken
        shown = trace != 0
        ratio = drive[shown] / trace[shown]
        np.testing.assert_allclose(ratio, np.full(ratio.shape, ratio[0] if len(ratio) else 0.0))


def test_a_lesson_and_the_dopamine_each_teach_the_gain():
    brain = routed()
    brain.recall.hearing = 0.5
    teach(brain)
    assert brain.carried_gains["read"] != 0.5
    assert 0.0 <= brain.carried_gains["read"] <= 1.0
    assert brain.describe()["memories"]["recall"]["hearing"] == brain.carried_gains["read"]
    brain = routed()
    brain.recall.hearing = 0.5
    live(brain)
    assert brain.carried_gains["read"] != 0.5
    assert 0.0 <= brain.carried_gains["read"] <= 1.0


def test_the_gain_reaches_no_higher_than_the_reads_amplitude():
    brain = routed()
    hearing = brain.hearing
    trace, read = hearing.routes
    assert read["bound"] == 1.0 and trace["bound"] == Hearing.BOUND
    for _ in range(2000):
        hearing.step(np.array([1.0, 1.0]))
    assert hearing.gain(read) == 1.0 and hearing.gain(trace) == Hearing.BOUND
    for _ in range(2000):
        hearing.step(np.array([-1.0, -1.0]))
    assert hearing.gain(read) == 0.0 and hearing.gain(trace) == 0.0
    # and lessons that would raise it leave it there
    brain = routed()
    teach(brain, lessons=24)
    assert 0.0 <= brain.carried_gains["read"] <= 1.0
    with pytest.raises((TypeError, ValueError), match="recall_drive_cap"):
        routed(recall_drive_cap=1.0)


def test_a_lesson_of_another_batch_reads_its_memory_as_silence():
    brain = routed()
    rng = np.random.default_rng(5)
    brain.act(rng.random((2, 6)), greedy=True)
    brain.reset()
    brain.lesson(rng.random((3, 6)), rng.integers(0, 4, size=3))
    assert 0.0 <= brain.carried_gains["read"] <= 1.0


def test_a_checkpoint_continues_exactly(tmp_path):
    brain = routed(history_decays=(0.5, 0.9))
    brain.recall.hearing = 0.5
    teach(brain)
    live(brain)
    rng = np.random.default_rng(9)
    # saved with an outcome pending: the act's contrast and the eligibility come back
    x = rng.random((2, 6))
    brain.act(x)
    copy = cd.Brain.load(brain.save(tmp_path / "brain.npz"))
    assert copy.carried_gains == brain.carried_gains
    assert copy.recall.hearing == brain.recall.hearing != 0.5
    assert copy.hearing.routes[-1]["bound"] == 1.0
    assert copy.describe()["memories"]["recall"]["hearing"] == brain.recall.hearing
    assert copy.describe()["genes"]["recall_drive_gain"] is True
    assert copy.hearing.routes[-1]["memory"] is copy.recall
    np.testing.assert_array_equal(copy.hearing.second, brain.hearing.second)
    assert copy.hearing.count == brain.hearing.count
    np.testing.assert_array_equal(
        copy.basal_ganglia.trace_hearing, brain.basal_ganglia.trace_hearing
    )
    np.testing.assert_array_equal(
        copy.basal_ganglia._pending_hearing, brain.basal_ganglia._pending_hearing
    )
    for t in range(6):
        reward, following = rng.random(2), rng.random((2, 6))
        done = np.full(2, t == 3)
        brain.learn(reward, done, following)
        copy.learn(reward, done, following)
        np.testing.assert_array_equal(brain.act(following), copy.act(following))
    assert copy.carried_gains == brain.carried_gains
    np.testing.assert_array_equal(brain.brain.efficacy, copy.brain.efficacy)


def test_a_checkpoint_continues_lessons_exactly(tmp_path):
    brain = routed()
    teach(brain)
    copy = cd.Brain.load(brain.save(tmp_path / "brain.npz"))
    rng = np.random.default_rng(10)
    for _ in range(4):
        x, label = rng.random((2, 6)), rng.integers(0, 4, size=2)
        np.testing.assert_array_equal(brain.step(x, teacher=label), copy.step(x, teacher=label))
    assert copy.carried_gains == brain.carried_gains
    np.testing.assert_array_equal(brain.brain.efficacy, copy.brain.efficacy)
    assert copy.parameters() == brain.parameters()


def test_a_file_with_the_route_and_no_gains_is_refused(tmp_path):
    import json

    path = routed().save(tmp_path / "brain.npz")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    saved = json.loads(str(arrays["brain"]))
    saved.pop("carried_gain")
    arrays["brain"] = np.array(json.dumps(saved))
    np.savez(tmp_path / "bare.npz", **arrays)
    with pytest.raises(ValueError, match="saved continuation state"):
        cd.Brain.load(tmp_path / "bare.npz")


def test_retune_and_grow_refuse_the_option():
    with pytest.raises(ValueError, match="construction setting"):
        routed().retune(recall_drive_gain=False)
    with pytest.raises(ValueError, match="construction setting"):
        brain_of().retune(recall_drive_gain=True)
    with pytest.raises(ValueError, match="recall_drive_gain"):
        brain_of().grow(recall_drive_gain=True)


def test_grow_carries_the_gain_over_new_heads():
    brain = routed()
    brain.recall.hearing = 0.5
    teach(brain)
    gains = brain.carried_gains
    assert gains["read"] != 0.5
    grown = brain.grow(recall_heads=2)
    assert grown.carried_gains == gains
    assert grown.hearing.count == brain.hearing.count
    k = list(gains).index("read")
    assert grown.hearing.second[k] == brain.hearing.second[k] > 0
    assert grown.hearing.routes[k]["width"] == grown.recall.neurons == 2 * brain.recall.neurons
    assert grown.basal_ganglia.hearing is grown.hearing
    assert grown.hearing.routes[k]["bound"] == 1.0
    assert grown.parameters() > brain.parameters()


def test_a_memory_grown_with_it_starts_its_route_at_one():
    brain = cd.Brain.compose(6, 4, modules=(16,), seed=5, recall=False, carried_gain=0.01)
    teach(brain)
    trace = brain.carried_gains["trace"]
    grown = brain.grow(recall=True, recall_founder=True, recall_drive_gain=True)
    assert grown.carried_gains == {"trace": trace, "read": 1.0}


def test_graft_keeps_this_brains_gains():
    brain, donor = routed(), routed()
    teach(brain)
    teach(donor, seed=8)
    gains = brain.carried_gains
    assert gains != donor.carried_gains
    grafted = brain.graft(donor, ("association", "motor"))
    assert grafted.carried_gains == gains
    assert grafted.hearing.routes[-1]["memory"] is grafted.recall


def test_an_imagined_branch_reads_at_the_gain_and_leaves_it():
    brain = routed()
    teach(brain)
    gains, second, count = brain.carried_gains, brain.hearing.second.copy(), brain.hearing.count
    rng = np.random.default_rng(5)
    brain.imagine([rng.random((2, 6)) for _ in range(2)], budget=256)
    assert brain.carried_gains == gains
    assert brain.hearing.count == count
    np.testing.assert_array_equal(brain.hearing.second, second)
    assert brain.recall.branch().hearing == brain.recall.hearing


def test_the_value_path_reads_the_port_as_it_is_held():
    tokens = (0, 1, 2, 3)
    brain = routed(recall_tokens=tokens, recall_gain=0.01)
    rng = np.random.default_rng(6)
    for _ in range(3):
        brain.act(rng.random((2, 6)), greedy=True)
    recall = brain.recall
    recall.gains[:] = 0.5
    x = rng.random((2, 6))
    motor = np.asarray(brain.motor_index)
    full = brain.stimulus(x)[:, motor]
    reads = recall._port_reads(recall._held_attention(recall.last))
    recall.hearing = 0.25
    quiet = brain.stimulus(x)[:, motor]
    np.testing.assert_allclose(quiet, 0.25 * full)
    held = recall._port_reads(recall._held_attention(recall.last))
    for loud, soft in zip(reads, held, strict=True):
        np.testing.assert_allclose(soft, 0.25 * loud)


@pytest.mark.parametrize(
    "options",
    [{"learned_code": 32}, {"recall_gain": 1.0, "recall_tokens": (0, 1, 2, 3)},
     {"recall_window": 3}, {"recall_next": False}],
    ids=["learned_code", "gain", "window", "no_next"],
)
def test_it_composes_with_the_memorys_other_forms(tmp_path, options):
    brain = routed(**options)
    assert brain.parameters() == gained(**options).parameters() + 1
    assert len(port_columns(brain)) == brain.recall.neurons
    brain.recall.hearing = 0.5
    teach(brain)
    assert brain.carried_gains["read"] != 0.5
    copy = cd.Brain.load(brain.save(tmp_path / "brain.npz"))
    assert copy.carried_gains == brain.carried_gains
    x = np.random.default_rng(3).random((2, 6))
    np.testing.assert_array_equal(brain.step(x), copy.step(x))


def test_a_memory_on_a_code_reads_on_its_ports_at_the_codes_width():
    brain = cd.Brain.compose(
        40, 4, modules=(16,), seed=5, code=64, recall=True, recall_founder=True, carried_gain=0.01, recall_drive_gain=True
    )
    assert len(port_columns(brain)) == 2 * 64 == brain.recall.neurons
    brain.recall.hearing = 0.5
    rng = np.random.default_rng(4)
    for _ in range(6):
        brain.step(rng.random((2, 40)), teacher=rng.integers(0, 4, size=2))
    assert brain.carried_gains["read"] != 0.5


@pytest.mark.parametrize(
    ("backend", "device", "precision", "atol"),
    [("torch", "cpu", "float64", 1e-9), ("torch", "mps", "float32", 1e-3),
     ("mlx", None, None, 1e-3)],
    ids=["torch_cpu64", "mps32", "mlx"],
)
def test_the_gain_learns_on_a_device_as_on_the_cpu(tmp_path, backend, device, precision, atol):
    if backend not in cd.available_backends():
        pytest.skip(f"{backend} not installed")
    if device == "mps":
        torch = pytest.importorskip("torch")
        if not torch.backends.mps.is_available():
            pytest.skip("MPS hardware unavailable")
    cpu = routed(recall_when=True)
    cpu.save(tmp_path / "brain.npz")
    moved = cd.Brain.load(
        tmp_path / "brain.npz", backend=backend, device=device, precision=precision
    )
    assert moved.brain.backend == backend
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for brain in (cpu, moved):
            brain.recall.hearing = 0.5
            teach(brain, lessons=6)
            brain.reset()
    assert cpu.carried_gains["read"] != 0.5
    np.testing.assert_allclose(
        list(moved.carried_gains.values()), list(cpu.carried_gains.values()), rtol=0, atol=atol
    )


def test_every_routine_keeps_the_gain_in_its_bounds():
    brain = routed(arousal=True, memory_choice=True, efference_amplitude=0.5, history_decays=(0.5,))
    rng = np.random.default_rng(4)
    x = rng.random((1, 6))
    brain.live(x)
    for _ in range(3):
        brain.live(x, reward=[1.0])
    brain.wait(x)
    brain.reset()
    brain.act(x)
    brain.learn(np.array([1.0]), np.array([False]), rng.random((1, 6)))
    brain.lesson(rng.random((1, 6)), np.array([1]))
    brain.imagine([rng.random((1, 6)) for _ in range(2)], budget=256)
    assert all(0.0 <= gain <= Hearing.BOUND for gain in brain.carried_gains.values())
    brain.reset()
    brain.fit(rng.random((8, 6)), rng.integers(0, 4, size=8), epochs=1, batch=4)
    assert all(0.0 <= gain <= Hearing.BOUND for gain in brain.carried_gains.values())
